# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Hugging Face Transformers library,
# specifically from the Mask2Former loss implementation, which itself is based on
# Mask2Former and DETR by Facebook, Inc. and its affiliates.
# Used under the Apache 2.0 License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerHungarianMatcher,
)


class MaskClassificationLoss(Mask2FormerLoss):
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
        # OE Params
        outlier_loss_enabled: bool = False,
        outlier_weight: float = 1.0,
        outlier_loss_target: str = "energy",
        outlier_loss_func: str = "squared_hinge",
        inlier_upper_threshold: float = -1.0,
        outlier_lower_threshold: float = -0.1,
        score_norm: str = "none",
    ):
        nn.Module.__init__(self)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.num_labels = num_labels
        self.eos_coef = no_object_coefficient
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.matcher = Mask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=class_coefficient,
        )

        # OE Params
        self.outlier_loss_enabled = outlier_loss_enabled
        self.outlier_weight = outlier_weight
        self.outlier_loss_target = outlier_loss_target
        self.outlier_loss_func = outlier_loss_func
        self.inlier_upper_threshold = inlier_upper_threshold
        self.outlier_lower_threshold = outlier_lower_threshold
        self.score_norm = score_norm

    @torch.compiler.disable
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict],
        class_queries_logits: Optional[torch.Tensor] = None,
    ):
        mask_labels = [
            target["masks"].to(masks_queries_logits.dtype) for target in targets
        ]
        class_labels = [target["labels"].long() for target in targets]

        indices = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
        )

        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices)
        loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)

        losses = {**loss_masks, **loss_classes}

        if self.outlier_loss_enabled:

            if any("outlier_masks" in t for t in targets):
                loss_outlier = self.outlier_loss(masks_queries_logits, class_queries_logits, targets)
                losses.update(loss_outlier)

        return losses

    def loss_masks(self, masks_queries_logits, mask_labels, indices):
        loss_masks = super().loss_masks(masks_queries_logits, mask_labels, indices, 1)

        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=masks_queries_logits.device
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        num_masks = torch.clamp(num_masks_tensor / world_size, min=1)

        for key in loss_masks.keys():
            loss_masks[key] = loss_masks[key] / num_masks

        return loss_masks

    def outlier_loss(self, mask_logits, class_logits, targets):
        """
        Adapted from RbA codebase
        """

        outlier_masks_list = []
        for x in targets:
            if "outlier_masks" in x:
                m = x["outlier_masks"]
                if m.dim() == 2:
                    m = m.unsqueeze(0)
                outlier_masks_list.append(m)
            else:

                pass
        
        if not outlier_masks_list:
            return {"outlier_loss": torch.tensor(0.0, device=mask_logits.device)}

        outlier_masks = torch.cat(outlier_masks_list, dim=0)

        ignore = (outlier_masks == 255)
        ood_mask = (outlier_masks == 1) & (~ignore)
        id_mask  = (outlier_masks == 0) & (~ignore)

        
        class_logits = F.softmax(class_logits, dim=-1)[..., :-1] 
        mask_logits = mask_logits.sigmoid()

        logits = torch.einsum("bqc,bqhw->bchw", class_logits, mask_logits)

        if self.outlier_loss_target == "nls":
            if self.score_norm == "sigmoid":
                score = logits.sigmoid()
            elif self.score_norm == "tanh":
                score = logits.tanh()
            else:
                score = logits
            score = -score.sum(dim=1) 
        elif self.outlier_loss_target == "energy":
            score = -torch.logsumexp(logits, dim=1) 
        elif self.outlier_loss_target == "softmax_entropy":
            score = torch.special.entr(logits.softmax(dim=1)).sum(dim=1)
        else:
             score = -torch.logsumexp(logits, dim=1)

 
        if score.shape[-2:] != outlier_masks.shape[-2:]:
             score = F.interpolate(score.unsqueeze(1), size=outlier_masks.shape[-2:], mode="bilinear", align_corners=True).squeeze(1)

        ood_score = score[ood_mask]
        id_score = score[id_mask]

        if self.outlier_loss_func == "squared_hinge":
            loss = torch.pow(
                F.relu(id_score - self.inlier_upper_threshold), 2).mean()
            if ood_mask.sum() > 0:
                loss = loss + \
                    torch.pow(
                        F.relu(self.outlier_lower_threshold - ood_score), 2).mean()
                loss = 0.5 * loss
        elif self.outlier_loss_func == "binary_cross_entropy":
  
             loss = 0.5 * F.binary_cross_entropy_with_logits(score, ood_mask.float())
        else:
             loss = torch.pow(
                F.relu(id_score - self.inlier_upper_threshold), 2).mean()
             if ood_mask.sum() > 0:
                loss = loss + \
                    torch.pow(
                        F.relu(self.outlier_lower_threshold - ood_score), 2).mean()
                loss = 0.5 * loss
         

        return {"outlier_loss": loss}


    def loss_total(self, losses_all_layers, log_fn) -> torch.Tensor:
        loss_total = None
        for loss_key, loss in losses_all_layers.items():
            log_fn(f"losses/train_{loss_key}", loss, sync_dist=True)

            if "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
            elif "cross_entropy" in loss_key:
                weighted_loss = loss * self.class_coefficient
            elif "outlier_loss" in loss_key:
                weighted_loss = loss * self.outlier_weight
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)

        log_fn("losses/train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total  

