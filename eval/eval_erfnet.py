# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from pathlib import Path
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3

NUM_CLASSES = 20
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default=["../Validation_Dataset/*/images/*.*"],
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    print("INPUT PATTERNS:", args.input)
    
    dataset_keys = {
        "RoadAnomaly21": "SMIYC_RA21",
        "RoadObsticle21": "SMIYC_RO21",
        "FS_LostFound_full": "FS_LF",
        "fs_static": "FS_Static",
        "RoadAnomaly": "RoadAnomaly", 
    }
    
    ood_gts_dict = {k: [] for k in dataset_keys.keys()}
    anomaly_score_dict = {
        k: {"MSP": [], "MaxLogit": [], "MaxEntropy": []}
        for k in dataset_keys.keys()
    }
    

    if not os.path.exists('results'):
        os.makedirs('results')
        
    results_path = os.path.join('results', 'results_ERFnet.txt')
    if not os.path.exists(results_path):
        open(results_path, 'w').close()
    file = open(results_path, 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict): 
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")
    model.eval()
    
    all_image_paths = []
    for pattern in args.input:
        all_image_paths.extend(glob.glob(os.path.expanduser(str(pattern))))

    def _belongs_to_dataset(path: str, dataset_id: str) -> bool:
        return dataset_id in Path(path).parts
    
    for path in all_image_paths:
        
        dataset_id = None
        for key in dataset_keys.keys():
            if _belongs_to_dataset(path, key):
                dataset_id = key
                break

        if dataset_id is None:
            continue
        
        if not os.path.isfile(path):
            continue
        
        ext = os.path.splitext(path)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            continue

        print(path)
        image = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()

        with torch.no_grad():
            result = model(image)
            
        logits = result.squeeze(0)
        
        #Compute MSP
        probs = torch.softmax(logits, dim=0)
        msp = probs.max(dim=0).values
        anomaly_msp = 1.0 - msp 
        
        #Compute MaxLogit
        maxlogit = logits.max(dim=0).values
        anomaly_maxlogit = - maxlogit
        
        # Compute MaxEntropy
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=0)  # (H, W) tensor
        anomaly_entropy = entropy
        
        #Convert the anomaly scores into numpy arrays(HxW)
        anomaly_msp = anomaly_msp.cpu().numpy()
        anomaly_maxlogit = anomaly_maxlogit.cpu().numpy()
        anomaly_entropy = anomaly_entropy.cpu().numpy()
        
        pathGT = path.replace("images", "labels_masks")         
        if dataset_id == "RoadObsticle21":
           pathGT = pathGT.replace("webp", "png")
        if dataset_id == "fs_static":
           pathGT = pathGT.replace("jpg", "png")                
        if dataset_id == "RoadAnomaly":
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        #Transform every value in the image into 0 (no anomaly), 1 (anomaly), 255 (pixel to ignore)
        if dataset_id == "RoadAnomaly":
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        #If in that image there is not at least an anomal pixel skip this image
        if 1 not in np.unique(ood_gts):
            del result, logits, probs, msp, maxlogit, entropy, anomaly_msp, anomaly_maxlogit, anomaly_entropy, ood_gts, mask
            continue
           

        ood_gts_dict[dataset_id].append(ood_gts)
        anomaly_score_dict[dataset_id]["MSP"].append(anomaly_msp)
        anomaly_score_dict[dataset_id]["MaxLogit"].append(anomaly_maxlogit)
        anomaly_score_dict[dataset_id]["MaxEntropy"].append(anomaly_entropy)
        
        
        #Delete local variables
        del result, logits, probs, msp, maxlogit, entropy, anomaly_msp, anomaly_maxlogit, anomaly_entropy, ood_gts, mask
        torch.cuda.empty_cache()

    file.write( "\n")
    file.write("----OOD EVALUATION ERFNet FOR DATASET (MSP / MaxLogit / MaxEntropy)----\n")

    for dataset_id, nice_name in dataset_keys.items():
        gts_list = ood_gts_dict[dataset_id]
        if len(gts_list) == 0:
            print(f"[{nice_name}] no images with anomalies, skipping dataset.")
            file.write(f"DATASET {nice_name}: NO SAMPLES (skipped)\n")
            continue

        ood_gts = np.array(gts_list)
        ood_mask = (ood_gts == 1)
        ind_mask = (ood_gts == 0)

        file.write(f"\nDATASET {nice_name}:\n")
        print(f"\n=== DATASET {nice_name} ===")

        for method_name, score_list in anomaly_score_dict[dataset_id].items():
            if len(score_list) == 0:
                print(f"  [{method_name}] no samples, skipping.")
                file.write(f"  Method {method_name}: NO SAMPLES (skipped)\n")
                continue

            anomaly_scores = np.array(score_list)

            ood_out = anomaly_scores[ood_mask]
            ind_out = anomaly_scores[ind_mask]

            ood_label = np.ones(len(ood_out))
            ind_label = np.zeros(len(ind_out))

            val_out = np.concatenate((ind_out, ood_out))
            val_label = np.concatenate((ind_label, ood_label))

            prc_auc = average_precision_score(val_label, val_out)
            fpr = fpr_at_95_tpr(val_out, val_label)

            print(f'  [{method_name}] AUPRC: {prc_auc * 100.0:.4f}   FPR@TPR95: {fpr * 100.0:.4f}')
            file.write(f'  Method {method_name}: AUPRC: {prc_auc * 100.0}   FPR@TPR95: {fpr * 100.0}\n')

    file.close()

if __name__ == '__main__':
    main()
