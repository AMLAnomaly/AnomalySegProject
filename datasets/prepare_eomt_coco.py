import os
import cv2
import shutil
import numpy as np
import random
import argparse
from pycocotools.coco import COCO
from tqdm import tqdm


DEFAULT_COCO_ROOT = "../coco/" 
DEFAULT_SPLIT = "train2017"
DEFAULT_MAX_IMAGES = 400
OOD_LABEL_VALUE = 254


CITYSCAPES_OVERLAP_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
    'parking meter', 'bench'
]

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare COCO masks for Outlier Exposure (Single-Object Strategy)")
    parser.add_argument("--root", type=str, default=DEFAULT_COCO_ROOT, help="Root directory of COCO dataset")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, help="COCO split (train2017 or val2017)")
    parser.add_argument("--max_images", type=int, default=DEFAULT_MAX_IMAGES, help="Maximum number of images to process")
    return parser.parse_args()

def main():
    args = parse_args()
    
    coco_root = args.root
    split = args.split
    max_images = args.max_images
    
    annotations_file = os.path.join(coco_root, "annotations", f"instances_{split}.json")
    source_img_dir = os.path.join(coco_root, split)
    
    output_mask_dir = os.path.join(coco_root, "annotations", f"ood_seg_{split}")

    output_img_dir = os.path.join(coco_root, f"{split}_ood_subset")

    if not os.path.exists(annotations_file):
        print(f"ERROR: file not found {annotations_file}")
        return

    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)

    coco = COCO(annotations_file)
    
    # Filter categories
    catIds = coco.getCatIds()
    cats = coco.loadCats(catIds)
    overlap_cat_ids = set([c['id'] for c in cats if c['name'] in CITYSCAPES_OVERLAP_CLASSES])
    
    #  Mix images
    all_imgIds = coco.getImgIds()
    
    random.seed(42) 
    random.shuffle(all_imgIds)
        
    count_saved = 0
    
    for imgId in tqdm(all_imgIds):
        if count_saved >= max_images:
            break

        annIds = coco.getAnnIds(imgIds=imgId)
        anns = coco.loadAnns(annIds)
        
        has_bad_class = False
        valid_anns = []

        for ann in anns:
            if ann['category_id'] in overlap_cat_ids:
                has_bad_class = True
                break # Discard image 
            if ann['iscrowd'] == 0 and 'segmentation' in ann:
                valid_anns.append(ann)
        
        if has_bad_class or not valid_anns:
            continue

        
        img_info = coco.loadImgs(imgId)[0]
        img_area = img_info['height'] * img_info['width']
        
        candidates = []
        for ann in valid_anns:
            if ann['area'] < 1000: 
                continue 
            if ann['area'] > (img_area * 0.40): 
                continue 
            candidates.append(ann)
            
        if not candidates:
            continue

        best_ann = max(candidates, key=lambda x: x['area'])

        img_info = coco.loadImgs(imgId)[0]
        
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
        ann_mask = coco.annToMask(best_ann)
        mask[ann_mask > 0] = OOD_LABEL_VALUE
            
        filename_png = os.path.splitext(img_info['file_name'])[0] + ".png"
        cv2.imwrite(os.path.join(output_mask_dir, filename_png), mask)
        
        filename_jpg = img_info['file_name']
        src_path = os.path.join(source_img_dir, filename_jpg)
        dst_path = os.path.join(output_img_dir, filename_jpg)
        
        try:
            shutil.copy(src_path, dst_path)
        except FileNotFoundError:
            continue
        
        count_saved += 1


if __name__ == "__main__":
    main()