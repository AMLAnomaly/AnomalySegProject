# Outlier Exposure for EoMT (Cityscapes)

This repository is based on  the original EoMT codebase. If something is unclear, refer to the upstream project: https://github.com/tue-mps/eomt.

The main addition here is **Outlier Exposure (OE)** for semantic segmentation: during training we paste COCO objects into Cityscapes images and train the model with an additional **outlier loss**.

## Repository map (most important files)

### Notebook (evaluation + fine-tuning)

- `OutlierExposureNotebook.ipynb`
  - Colab notebook that shows:
    - How to run EoMT OOD evaluation (`python eval/eval_eomt.py`)
    - How to compute EoMT Cityscapes mIoU (`python eval/eval_eomt_iou.py`)
    - How to run ERFNet OOD evaluation (`python eval/eval_erfnet.py`)
    - How to compute ERFNet Cityscapes mIoU (`python eval/eval_erfnet_iou.py`)
    - How to start (and resume) **fine-tuning EoMT with OE** using `python main.py fit` and the config `configs/dinov2/cityscapes/semantic/eomt_base_640_oe.yaml`.

### Configuration

- `configs/dinov2/cityscapes/semantic/eomt_base_640_oe.yaml`
  - Main config used for **fine-tuning EoMT with Outlier Exposure**.
  - Key parts:
    - `model.init_args.outlier_loss_enabled`, `outlier_weight`, thresholds (`inlier_upper_threshold`, `outlier_lower_threshold`).
    - `data.init_args.oe_enabled` and `ood_prob`.
    - Optional OE strategies: `oe_road_only` (paste only on road/sidewalk) and `oe_perspective_scale`.
    - W&B logging settings are in `trainer.logger`.

### COCO utilities and preparation (OE source)

- `datasets/coco_utils.py`
  - `COCO` dataset wrapper used to load **OOD masks** and the corresponding **COCO images**.
  - Reads masks from `annotations/ood_seg_<split><year>/` (e.g., `ood_seg_train2017`) and matches them to COCO JPG images.
  - Provides helper utilities for object extraction/mixing (e.g., `extract_bboxes`, `mix_object`).

- `datasets/prepare_eomt_coco.py`
  - Script that creates **one-object OOD masks** for COCO (label `254`) and copies a subset of COCO images.
  - It filters out COCO categories that overlap with Cityscapes classes (to reduce “inlier-looking” pasted objects).
  - Outputs:
    - `annotations/ood_seg_<split>/` (binary-ish mask PNGs with value `254`)
    - `<split>_ood_subset/` (matching COCO images)

### Data transforms (where OE happens)

- `datasets/transforms.py`
  - Contains the `Transforms` module used by datasets.
  - The key method is `Transforms.mix_ood(...)`:
    - With probability `ood_prob`, samples an OOD object/mask from the COCO OE dataset.
    - Crops the object tightly using its bounding box.
    - Pastes the object into the Cityscapes image (optionally road/sidewalk only; optionally perspective-scaled).
    - Produces an `outlier_mask` (0 = inlier, 1 = OOD, 255 = ignore) and updates targets so that inlier masks don’t “cover” pasted OOD pixels.

### Training loss (where OE is learned)

- `training/mask_classification_loss.py`
  - Defines `MaskClassificationLoss`, extending HuggingFace Mask2Former loss.
  - Adds an OE-specific `outlier_loss(...)` term when `outlier_loss_enabled=True` and targets include `outlier_masks`.
  - Supported scoring targets include:
    - `nls` (negative logit sum)
  - Loss type can be `squared_hinge` (default) or `binary_cross_entropy`.
  - The OE loss is weighted by `outlier_weight` in `loss_total`.

### Evaluation code (EoMT and ERFNet)

All evaluation scripts below write text outputs into a local `results/` folder.
If you run them from inside `eval/` (as done in the notebook), outputs end up in `eval/results/`.

- `eval/results/`
  - Stores evaluation outputs (e.g., `results_EoMT.txt`, `results_ERFnet.txt`, temperature scaling logs, etc.).
  - Common files you may see here:
    - `results_EoMT.txt`, `results_ERFnet.txt` (mIoU and/or OOD logs depending on script)
    - `results_EoMT_Temperature_cl.txt`, `results_EoMT_Temperature_pl.txt`
    - `results_EoMT_Temperature_mIoU_cl.txt`
    - `results_EoMT_Outlier_Exposure.txt` (OE experiment logs)

- `eval/eomt_utils.py`
  - Utility entry point used by several evaluation scripts.
  - Builds a ready-to-evaluate Lightning model via `build_eomt_cityscapes_lit_model_for_eval(...)`.

- `eval/eval_eomt.py`
  - OOD evaluation for EoMT on the 5 benchmark datasets (FS Lost&Found, FS Static, SMIYC RO21, SMIYC RA21, Road Anomaly).
  - Computes multiple pixel-level anomaly scores (MSP, MaxLogit, MaxEntropy, RbA) and logs metrics such as AUPRC and FPR@TPR95.

- `eval/eval_eomt_iou.py`
  - Computes **mIoU on Cityscapes** for EoMT.
  - Appends mIoU results into `results/results_EoMT.txt`.

- `eval/eval_eomt_temperature_scaling_cl.py`
  - OOD evaluation with **temperature scaling applied to class logits** before converting to per-pixel logits.
  - Writes to `results/results_EoMT_Temperature_cl.txt`.

- `eval/eval_eomt_temperature_scaling_pl.py`
  - OOD evaluation with **temperature scaling applied on per-pixel logits**.
  - Writes to `results/results_EoMT_Temperature_pl.txt`.

- `eval/eval_eomt_miou_temperature_scaling_cl.py`
  - Computes Cityscapes mIoU for a list of temperatures (class-logit scaling).
  - Writes to `results/results_EoMT_Temperature_mIoU_cl.txt`.

- `eval/eval_erfnet.py`
  - OOD evaluation for ERFNet (MSP / MaxLogit / MaxEntropy) on the same benchmark datasets.
  - Writes to `results/results_ERFnet.txt`.

- `eval/eval_erfnet_iou.py`
  - Computes **mIoU on Cityscapes** for ERFNet.
  - Appends mIoU results into `results/results_ERFnet.txt`.



