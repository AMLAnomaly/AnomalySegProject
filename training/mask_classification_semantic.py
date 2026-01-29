# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch.nn as nn
import torch.nn.functional as F

from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule
from sklearn.metrics import average_precision_score
from ood_metrics import fpr_at_95_tpr
import torch
import numpy as np
import gc


class MaskClassificationSemantic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        ignore_idx: int = 255,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
        # OE Params
        outlier_loss_enabled: bool = False,
        outlier_weight: float = 1.0,
        outlier_loss_target: str = "energy",
        outlier_loss_func: str = "squared_hinge",
        inlier_upper_threshold: float = -1.0,
        outlier_lower_threshold: float = -0.1,
        score_norm: str = "none",
        freeze_encoder: bool = True,
        num_unfrozen_encoder_blocks: int = 0,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
            outlier_loss_enabled=outlier_loss_enabled,
            outlier_weight=outlier_weight,
            outlier_loss_target=outlier_loss_target,
            outlier_loss_func=outlier_loss_func,
            inlier_upper_threshold=inlier_upper_threshold,
            outlier_lower_threshold=outlier_lower_threshold,
            score_norm=score_norm,
            freeze_encoder=freeze_encoder,
            num_unfrozen_encoder_blocks=num_unfrozen_encoder_blocks,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.ignore_idx = ignore_idx
        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes = range(num_classes)

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
            outlier_loss_enabled=outlier_loss_enabled,
            outlier_weight=outlier_weight,
            outlier_loss_target=outlier_loss_target,
            outlier_loss_func=outlier_loss_func,
            inlier_upper_threshold=inlier_upper_threshold,
            outlier_lower_threshold=outlier_lower_threshold,
            score_norm=score_norm,
        )

        self.init_metrics_semantic(ignore_idx, self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1)
        
        self.ood_scores = []
        self.ood_gts = []

    def compute_rba_score(self, pixel_logits):

        class_probs = torch.tanh(pixel_logits)
        known_class_prob_sum = class_probs.sum(dim=1)
        rba_score = -known_class_prob_sum
        return rba_score

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            return self.eval_step(batch, batch_idx, "val")
        elif dataloader_idx == 1:
            return self.eval_ood_step(batch, batch_idx)

    def eval_ood_step(self, batch, batch_idx):
        imgs, targets = batch 
        
        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = self.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer = self(crops)

        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], self.img_size, mode="bilinear", align_corners=False
        )
        
        crop_logits = self.to_per_pixel_logits_semantic(
            mask_logits, class_logits_per_layer[-1]
        )
        
        pixel_logits_list = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)
        
        for i, pixel_logits in enumerate(pixel_logits_list):

            rba_score = self.compute_rba_score(pixel_logits.unsqueeze(0)).squeeze(0) 


            gt = targets[i].detach().cpu().numpy().astype(np.uint8)
            score = rba_score.detach().cpu().numpy()
            

            if 1 not in np.unique(gt):
                continue

            self.ood_gts.append(gt)
            self.ood_scores.append(score)

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = self.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer = self(crops)

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            crop_logits = self.to_per_pixel_logits_semantic(mask_logits, class_logits)
            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            self.update_metrics_semantic(logits, targets, i)

            if batch_idx == 0:
                self.plot_semantic(
                    imgs[0], targets[0], logits[0], log_prefix, i, batch_idx
                )

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_semantic("val")
        
        if len(self.ood_gts) > 0:
            print("\nComputing OOD Metrics...")

            
            try:

                ood_gts_flat = np.concatenate([g.flatten() for g in self.ood_gts]).astype(np.uint8)
                scores_flat = np.concatenate([s.flatten() for s in self.ood_scores]).astype(np.float32)

                ood_mask = (ood_gts_flat == 1)
                ind_mask = (ood_gts_flat == 0)

                ood_out = scores_flat[ood_mask]
                ind_out = scores_flat[ind_mask]

                ood_label = np.ones(len(ood_out), dtype=np.uint8)
                ind_label = np.zeros(len(ind_out), dtype=np.uint8)

                val_out = np.concatenate((ind_out, ood_out))
                val_label = np.concatenate((ind_label, ood_label))

                ap = average_precision_score(val_label, val_out)
                fpr = fpr_at_95_tpr(val_out, val_label)
                
                self.log("val_ood_ap", ap, prog_bar=True)
                self.log("val_ood_fpr95", fpr, prog_bar=True)
                
                print(f"RoadAnomaly Validation: AP={ap*100:.2f}, FPR95={fpr*100:.2f}")
                
            except Exception as e:
                print(f"Error computing OOD metrics: {e}")
            
            self.ood_gts.clear()
            self.ood_scores.clear()
            gc.collect()

    def on_validation_end(self):
        self._on_eval_end_semantic("val")
