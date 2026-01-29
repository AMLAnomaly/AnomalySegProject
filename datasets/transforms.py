# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from Detectron2 by Facebook, Inc. and its affiliates,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

import torch
import numpy as np
import random
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import wrap, TVTensor, Mask
from torch import nn, Tensor
from typing import Any, Union, Optional
from datasets.coco_utils import COCO, mix_object


class Transforms(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        color_jitter_enabled: bool,
        scale_range: tuple[float, float],
        max_brightness_delta: int = 32,
        max_contrast_factor: float = 0.5,
        saturation_factor: float = 0.5,
        max_hue_delta: int = 18,
        oe_enabled: bool = False,
        coco_root: str = "datasets/coco",
        ood_prob: float = 0.2,
        ood_label: int = 254,
        ignore_label: int = 255, 
        coco_split: str = "train",
        oe_road_only: bool = False,
        oe_perspective_scale: bool = False,
    ):
        super().__init__()

        self.img_size = img_size
        self.color_jitter_enabled = color_jitter_enabled
        self.max_brightness_factor = max_brightness_delta / 255.0
        self.max_contrast_factor = max_contrast_factor
        self.max_saturation_factor = saturation_factor
        self.max_hue_delta = max_hue_delta / 360.0

        self.random_horizontal_flip = T.RandomHorizontalFlip()
        self.scale_jitter = T.ScaleJitter(target_size=img_size, scale_range=scale_range)
        self.random_crop = T.RandomCrop(img_size)

        # OE Params
        self.oe_enabled = oe_enabled
        self.ood_prob = ood_prob
        self.ood_label = ood_label
        self.ignore_label = ignore_label
        self.oe_road_only = oe_road_only
        self.oe_perspective_scale = oe_perspective_scale
        
        self.coco_dataset = None
        if self.oe_enabled:
            print(f"Initializing COCO dataset for OE from {coco_root} (split: {coco_split})...")
            if self.oe_road_only:
                print(" -> OE Strategy: Region-Aware (Road/Sidewalk only) enabled")
            if self.oe_perspective_scale:
                print(" -> OE Strategy: Perspective Scaling enabled")
            self.coco_dataset = COCO(root=coco_root, proxy_size=300, split=coco_split)

    def _random_factor(self, factor: float, center: float = 1.0):
        return torch.empty(1).uniform_(center - factor, center + factor).item()

    def _brightness(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_brightness(
                img, self._random_factor(self.max_brightness_factor)
            )

        return img

    def _contrast(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_contrast(img, self._random_factor(self.max_contrast_factor))

        return img

    def _saturation_and_hue(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_saturation(
                img, self._random_factor(self.max_saturation_factor)
            )

        if torch.rand(()) < 0.5:
            img = F.adjust_hue(img, self._random_factor(self.max_hue_delta, center=0.0))

        return img

    def color_jitter(self, img):
        if not self.color_jitter_enabled:
            return img

        img = self._brightness(img)

        if torch.rand(()) < 0.5:
            img = self._contrast(img)
            img = self._saturation_and_hue(img)
        else:
            img = self._saturation_and_hue(img)
            img = self._contrast(img)

        return img

    def pad(
        self, img: Tensor, target: dict[str, Any]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        pad_h = max(0, self.img_size[-2] - img.shape[-2])
        pad_w = max(0, self.img_size[-1] - img.shape[-1])

        
        padding = [0, 0, pad_w, pad_h]

        img = F.pad(img, padding)
        target["masks"] = F.pad(target["masks"], padding)
        if "outlier_masks" in target:
             target["outlier_masks"] = F.pad(target["outlier_masks"], padding)

        return img, target

    def _filter(self, target: dict[str, Union[Tensor, TVTensor]], keep: Tensor) -> dict:
        result = {k: wrap(v[keep], like=v) for k, v in target.items() if k != "outlier_masks"}
        if "outlier_masks" in target:
            result["outlier_masks"] = target["outlier_masks"]
        return result

    def mix_ood(self, img: Tensor, target: dict[str, Any]):
        """
        Mixes OOD object into the image and updates targets.
        Img is (C, H, W).
        """
        if self.coco_dataset is None or len(self.coco_dataset) == 0:
             return img, target, None

        if np.random.rand() >= self.ood_prob:
            return img, target, None

        img_np = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        # Reconstruct semantic mask
        h, w = img.shape[-2:]
        sem_seg_gt = torch.full((h, w), self.ignore_label, dtype=torch.long, device=img.device)
        for i, mask in enumerate(target["masks"]):
            sem_seg_gt[mask.bool()] = target["labels"][i] 
        sem_seg_gt_np = sem_seg_gt.cpu().numpy()

        # Get OOD Object
        ood_idx = np.random.randint(0, len(self.coco_dataset))
        ood_object, ood_mask = self.coco_dataset[ood_idx]
        

        from datasets.coco_utils import extract_bboxes 
        ood_object_np = np.array(ood_object)
        ood_mask_np = np.array(ood_mask)
        
        mask_boolean = (ood_mask_np == self.ood_label)
        if mask_boolean.sum() == 0:
             return img, target, None 

        # Extract bbox to crop tight
        ood_boxes = extract_bboxes(np.expand_dims(mask_boolean, axis=2))
        if ood_boxes.shape[0] == 0:
             return img, target, None
             
        y1, x1, y2, x2 = ood_boxes[0]
        # Crop tight
        cut_img = ood_object_np[y1:y2, x1:x2, :]
        cut_mask = ood_mask_np[y1:y2, x1:x2]
        
        # 1. Perspective Scaling
        if self.oe_perspective_scale:
            pass 
            
        
        # Try to find a valid placement
        max_attempts = 30
        for attempt in range(max_attempts):
            h_img, w_img = sem_seg_gt_np.shape
            
            # Default Random
            v_y = np.random.randint(0, h_img)
            v_x = np.random.randint(0, w_img)
            

            current_cut_img = cut_img
            current_cut_mask = cut_mask
            
            if self.oe_perspective_scale:

                scale = 0.3 + 0.9 * (v_y / h_img)
                
                from PIL import Image
                pil_img = Image.fromarray(cut_img)
                pil_mask = Image.fromarray(cut_mask)
                
                new_w = int(cut_img.shape[1] * scale)
                new_h = int(cut_img.shape[0] * scale)
                
                if new_w < 1 or new_h < 1: continue 
                
                current_cut_img = np.array(pil_img.resize((new_w, new_h), Image.BILINEAR))
                current_cut_mask = np.array(pil_mask.resize((new_w, new_h), Image.NEAREST))
            
            h_obj, w_obj = current_cut_img.shape[:2]
            
            # check if it fits in image
            if v_y + h_obj > h_img or v_x + w_obj > w_img:
                continue

            roi_gt = sem_seg_gt_np[v_y:v_y+h_obj, v_x:v_x+w_obj]

            obj_mask_bool = (current_cut_mask == self.ood_label)

            underlying_classes = roi_gt[obj_mask_bool]
            if underlying_classes.size == 0:
                continue
            if np.any(underlying_classes == self.ignore_label):
                continue

            if self.oe_road_only:

                valid_pixels = np.isin(underlying_classes, [0, 1])
                if valid_pixels.mean() < 0.5:
                     continue


            result_image = img_np.copy()
            roi_image = result_image[v_y:v_y+h_obj, v_x:v_x+w_obj]
            mask_bool = (current_cut_mask == self.ood_label)
            roi_image[mask_bool] = current_cut_img[mask_bool]
            result_image[v_y:v_y+h_obj, v_x:v_x+w_obj] = roi_image
            

            outlier_mask_np = np.zeros((h_img, w_img), dtype=np.uint8)
            roi_outlier = outlier_mask_np[v_y:v_y+h_obj, v_x:v_x+w_obj]
            roi_outlier[mask_bool] = 1 # 1 means OOD
            outlier_mask_np[v_y:v_y+h_obj, v_x:v_x+w_obj] = roi_outlier
            

            outlier_mask_np[sem_seg_gt_np == self.ignore_label] = self.ignore_label
            
            mixed_img = torch.from_numpy(result_image).permute(2, 0, 1).to(img.device)
            outlier_mask = torch.from_numpy(outlier_mask_np).long().to(img.device)
            
            ood_pixels = (outlier_mask == 1)
            target["masks"] = target["masks"] & (~ood_pixels)

            return mixed_img, target, outlier_mask

        return img, target, None

    def forward(
        self, img: Tensor, target: dict[str, Union[Tensor, TVTensor]]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        img_orig, target_orig = img, target

        target = self._filter(target, ~target["is_crowd"])

        img = self.color_jitter(img)
        img, target = self.random_horizontal_flip(img, target)
        img, target = self.scale_jitter(img, target)
        img, target = self.pad(img, target)
        img, target = self.random_crop(img, target)

        if self.oe_enabled:
            img, target, outlier_mask = self.mix_ood(img, target)
            if outlier_mask is not None:
                target["outlier_masks"] = Mask(outlier_mask.unsqueeze(0))
            else:
                target["outlier_masks"] = Mask(
                    torch.zeros((1, *img.shape[-2:]), dtype=torch.long, device=img.device)
                )

        valid = target["masks"].flatten(1).any(1)
        if not valid.any():
            return self(img_orig, target_orig)

        target = self._filter(target, valid)
        
        return img, target

