# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import glob
import torch
import random
import gc
from PIL import Image
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from sklearn.metrics import average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor
import torch.nn.functional as F
from skimage.measure import block_reduce
from pathlib import Path

from ood_metrics import fpr_at_95_tpr

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
eomt_root = os.path.join(project_root, "eomt")

for p in [project_root, eomt_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from eomt_utils import (
    build_eomt_cityscapes_lit_model_for_eval
)


seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3

NUM_CLASSES = 19

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# DOWNSAMPLING FACTOR FOR MEMORY EFFICIENCY
# Factor 2: reduces 2048x1024 -> 1024x512 (4x less memory)

def compute_rba_score(pixel_logits):

    class_probs = torch.tanh(pixel_logits)  
    
    known_class_prob_sum = class_probs.sum(dim=1) 
    
    rba_score = -known_class_prob_sum
    
    return rba_score

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default=["../Validation_Dataset/*/images/*.*"],
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="epoch_106-step_19902_eomt.ckpt")
    parser.add_argument('--subset', default="val")
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--downsample', type=int, default=2, 
                        help='Downsample factor for memory efficiency')
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else "cuda")
    
    downsample_factor = args.downsample

    print("INPUT PATTERNS:", args.input)
    print(f"DOWNSAMPLE FACTOR: {downsample_factor} (memory reduction: {downsample_factor**2}x)")

    dataset_keys = {
        "RoadAnomaly21": "SMIYC_RA21",
        "RoadObsticle21": "SMIYC_RO21",
        "FS_LostFound_full": "FS_LF",
        "fs_static": "FS_Static",
        "RoadAnomaly": "RoadAnomaly",
    }

    if not os.path.exists('results'):
        os.makedirs('results')
        
    results_path = os.path.join('results', 'results_EoMT_Temperature_cl.txt')
    if not os.path.exists(results_path):
        open(results_path, 'w').close()
    file = open(results_path, 'a')

    weightspath = osp.join(args.loadDir, args.loadWeights)
    print("Loading EoMT checkpoint:", weightspath)

    lit_model= build_eomt_cityscapes_lit_model_for_eval(weightspath, device, cpu=args.cpu)
    
    print("EoMT model loaded successfully.")
    
    temperatures = [0.5, 0.75, 1.0, 1.1]

    all_image_paths_raw = []
    for pattern in args.input:
        all_image_paths_raw.extend(glob.glob(os.path.expanduser(str(pattern))))

    def _belongs_to_dataset(path: str, dataset_id: str) -> bool:
        return dataset_id in Path(path).parts

    file.write("\n")
    file.write(f"----OOD EVALUATION EoMT Temperature Scaling (downsample={downsample_factor})----\n")

    for dataset_id, nice_name in dataset_keys.items():
        print(f"\n=== PROCESSING DATASET: {nice_name} ===")
        
        current_dataset_paths = [p for p in all_image_paths_raw if _belongs_to_dataset(p, dataset_id)]
        
        if len(current_dataset_paths) == 0:
            print(f"No images found for {nice_name}")
            file.write(f"DATASET {nice_name}: NO SAMPLES (skipped)\n")
            continue

        current_gts = []
        current_scores = {f"MSP(t={t})": [] for t in temperatures}

        for path in current_dataset_paths:
            if not os.path.isfile(path):
                continue

            ext = os.path.splitext(path)[1].lower()
            if ext not in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
                continue

            print(path)

            img = Image.open(path).convert('RGB')
            img_np = np.array(img)
            del img
            
            images = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            del img_np
            
            if not args.cpu:
                images = images.to(device)
            img_sizes = [img.shape[-2:] for img in images]

            with torch.no_grad():
                crops, origins = lit_model.window_imgs_semantic(images) 
                mask_logits_per_layer, class_logits_per_layer = lit_model(crops)

                mask_logits = F.interpolate(
                mask_logits_per_layer[-1], lit_model.img_size, mode="bilinear",align_corners=False
                )

                class_logits_last = class_logits_per_layer[-1]

            for t in temperatures:
                scaled_class_logits = class_logits_last / t

                crop_logits = lit_model.to_per_pixel_logits_semantic(
                    mask_logits, scaled_class_logits
                )

                pixel_logits_list = lit_model.revert_window_logits_semantic(
                    crop_logits, origins, img_sizes
                )

                logits = pixel_logits_list[0].squeeze(0)  # [C, H, W]

                probs = torch.softmax(logits, dim=0)
                msp = probs.max(dim=0).values
                anomaly_msp = 1.0 - msp
                
                anomaly_msp_np = anomaly_msp.cpu().numpy()
                del probs, msp, anomaly_msp, logits, crop_logits, pixel_logits_list, scaled_class_logits

                if downsample_factor > 1:
                    anomaly_msp_np = block_reduce(anomaly_msp_np, (downsample_factor, downsample_factor), np.mean)
                
                current_scores[f"MSP(t={t})"].append(anomaly_msp_np)
            del mask_logits, class_logits_last

            # Path GT
            pathGT = path.replace("images", "labels_masks")
            if dataset_id == "RoadObsticle21":
                pathGT = pathGT.replace("webp", "png")
            if dataset_id == "fs_static":
                pathGT = pathGT.replace("jpg", "png")
            if dataset_id == "RoadAnomaly":
                pathGT = pathGT.replace("jpg", "png")

            mask = Image.open(pathGT)
            ood_gts = np.array(mask)
            del mask

            if dataset_id == "RoadAnomaly":
                ood_gts = np.where((ood_gts == 2), 1, ood_gts)
            if "LostAndFound" in pathGT:
                ood_gts = np.where((ood_gts == 0), 255, ood_gts)
                ood_gts = np.where((ood_gts == 1), 0, ood_gts)
                ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
            if "Streethazard" in pathGT:
                ood_gts = np.where((ood_gts == 14), 255, ood_gts)
                ood_gts = np.where((ood_gts < 20), 0, ood_gts)
                ood_gts = np.where((ood_gts == 255), 1, ood_gts)


            if downsample_factor > 1:
                ood_gts = block_reduce(ood_gts.astype(np.float32), (downsample_factor, downsample_factor), np.max).astype(np.uint8)

            if 1 not in np.unique(ood_gts):
                for t in temperatures:
                    current_scores[f"MSP(t={t})"].pop()
                torch.cuda.empty_cache()
                gc.collect()
                continue

           
            
            current_gts.append(ood_gts)
            
            torch.cuda.empty_cache()
            gc.collect()

        if len(current_gts) > 0:
            file.write(f"\nDATASET {nice_name}:\n")
            print(f"\n=== DATASET {nice_name} ===")
            


            for method_name, score_list in current_scores.items():
                if len(score_list) == 0:
                    print(f"  [{method_name}] no samples, skipping.")
                    file.write(f"  Method {method_name}: NO SAMPLES (skipped)\n")
                    continue

                anomaly_scores = np.concatenate([s.flatten() for s in score_list])

                ood_gts = np.concatenate([g.flatten() for g in current_gts])
                ood_mask = (ood_gts == 1)
                ind_mask = (ood_gts == 0)

                del ood_gts

                ood_out = anomaly_scores[ood_mask]
                ind_out = anomaly_scores[ind_mask]

                del anomaly_scores
                gc.collect()
                
                ood_label = np.ones(len(ood_out))
                ind_label = np.zeros(len(ind_out))

                val_out = np.concatenate((ind_out, ood_out))
                val_label = np.concatenate((ind_label, ood_label))

                del ood_out, ind_out, ood_label, ind_label
                gc.collect()

                prc_auc = average_precision_score(val_label, val_out)
                fpr = fpr_at_95_tpr(val_out, val_label)

                print(
                    f'  [{method_name}] AUPRC: {prc_auc * 100.0:.4f}   FPR@TPR95: {fpr * 100.0:.4f}'
                )
                file.write(
                    f'  Method {method_name}: AUPRC: {prc_auc * 100.0}   FPR@TPR95: {fpr * 100.0}\n'
                )

                del val_out, val_label
                gc.collect()
        else:
            print(f"[{nice_name}] no images with anomalies, skipping dataset.")
            file.write(f"DATASET {nice_name}: NO SAMPLES (skipped)\n")

        del current_gts
        del current_scores
        if 'ood_gts' in locals(): del ood_gts
        if 'anomaly_scores' in locals(): del anomaly_scores
        gc.collect()
        print(f"Memory cleared for {nice_name}")

    file.close()


if __name__ == '__main__':
    main()