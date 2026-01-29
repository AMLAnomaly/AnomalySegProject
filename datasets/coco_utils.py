# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the RbA repository
# ---------------------------------------------------------------

import os
import random
import numpy as np
from typing import Optional, Callable
from torch.utils.data import Dataset
from PIL import Image

class COCO(Dataset):
    train_id_in = 0
    train_id_out = 254
    min_image_size = 480

    def __init__(self, root: str, proxy_size: int, split: str = "train",
                 transform: Optional[Callable] = None, shuffle=True) -> None:
        """
        COCO dataset loader
        """
        self.root = root
        self.coco_year = '2017'
        self.split = split + self.coco_year
        self.images = []
        self.targets = []
        self.transform = transform
        
        annotation_dir = os.path.join(self.root, "annotations", "ood_seg_" + self.split)
        
        if not os.path.exists(annotation_dir):
            print(f"Warning: COCO annotation dir not found: {annotation_dir}")

        for root_dir, _, filenames in os.walk(annotation_dir):
            for filename in filenames:
                if os.path.splitext(filename)[-1] == '.png':
                    self.targets.append(os.path.join(root_dir, filename))
                    self.images.append(os.path.join(self.root, self.split, filename.split(".")[0] + ".jpg"))



        if shuffle and len(self.images) > 0:
            zipped = list(zip(self.images, self.targets))
            random.shuffle(zipped)
            self.images, self.targets = zip(*zipped)

        if proxy_size is not None and len(self.images) > 0:
            self.images = list(self.images[:int(proxy_size)])
            self.targets = list(self.targets[:int(proxy_size)])
        elif len(self.images) > 5000:
            self.images = list(self.images[:5000])
            self.targets = list(self.targets[:5000])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = Image.open(self.images[i]).convert('RGB')
        target = Image.open(self.targets[i]).convert('L')
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].

    Adapted from RbA """

    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            x2 += 1
            y2 += 1
        else:

            x1, x2, y1, y2 = 0, 0, 0, 0

        boxes[i] = np.array([y1, x1, y2, x2])

    return boxes.astype(np.int32)


def mix_object(current_labeled_image, current_labeled_mask, cut_object_image, cut_object_mask, ood_label):
    """
    Adapted from RbA
    
    current_labeled_image: numpy array (H, W, 3)
    current_labeled_mask: numpy array (H, W)
    cut_object_image: numpy array/PIL
    cut_object_mask: numpy array/PIL
    ood_label: int
    """
    current_labeled_image = np.array(current_labeled_image)
    current_labeled_mask = np.array(current_labeled_mask)
    cut_object_image = np.array(cut_object_image)
    cut_object_mask = np.array(cut_object_mask)

    mask = (cut_object_mask == ood_label)
    
    ood_mask = np.expand_dims(mask, axis=2)
    ood_boxes = extract_bboxes(ood_mask)
    if ood_boxes.shape[0] == 0:
         return current_labeled_image, current_labeled_mask

    ood_boxes = ood_boxes[0, :]
    y1, x1, y2, x2 = ood_boxes[0], ood_boxes[1], ood_boxes[2], ood_boxes[3]
    cut_object_mask_cropped = cut_object_mask[y1:y2, x1:x2]
    cut_object_image_cropped = cut_object_image[y1:y2, x1:x2, :]

    mask_cropped = cut_object_mask_cropped == ood_label

    h_img, w_img = current_labeled_mask.shape[:2]
    h_cut, w_cut = cut_object_mask_cropped.shape[:2]

    if mask_cropped.sum() > 0:
        if h_img - h_cut < 0 or w_img - w_cut < 0:

            return current_labeled_image, current_labeled_mask
            
        h_start_point = random.randint(0, h_img - h_cut)
        h_end_point = h_start_point + h_cut
        w_start_point = random.randint(0, w_img - w_cut)
        w_end_point = w_start_point + w_cut
    else:
        return current_labeled_image, current_labeled_mask
    
    result_image = current_labeled_image.copy()
    
    # Paste image
    roi_image = result_image[h_start_point:h_end_point, w_start_point:w_end_point, :]
    roi_image[mask_cropped] = cut_object_image_cropped[mask_cropped]
    result_image[h_start_point:h_end_point, w_start_point:w_end_point, :] = roi_image
    
    # Paste mask
    result_label = current_labeled_mask.copy()
    roi_label = result_label[h_start_point:h_end_point, w_start_point:w_end_point]
    roi_label[mask_cropped] = cut_object_mask_cropped[mask_cropped]
    result_label[h_start_point:h_end_point, w_start_point:w_end_point] = roi_label

    return result_image, result_label
