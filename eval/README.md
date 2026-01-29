# Anomaly Segmentation Eval

In this folder you can find scripts to evaluate your model's output.

For anomaly segmentation metrics on the validation datasets, the main entry points are:

- `eval_erfnet.py` (ERFNet)
- `eval_eomt.py` (EoMT)

For semantic segmentation IoU on Cityscapes, use:

- `eval_erfnet_iou.py` (ERFNet)
- `eval_eomt_iou.py` (EoMT)

## Requirements

It could work with the default runtime of Colab or other versions of the libraries but these are the requirements this code was tested on.

- [**Python**](https://www.python.org/)
- [**PyTorch**](http://pytorch.org/)
- **Additional Python packages**: numpy, matplotlib, Pillow, torchvision and visdom (optional for --visualize flag)


## Anomaly Inference

The scripts `eval_erfnet.py` and `eval_eomt.py` accept one or more `--input` glob patterns.

Default input pattern is:
`../Validation_Dataset/*/images/*.*`


## Functions for evaluating/visualizing the network's output

Currently there are these usable scripts:

- eval_erfnet
- eval_eomt
- eval_eomt_temperature_scaling_cl
- eval_eomt_temperature_scaling_pl
- eval_erfnet_iou
- eval_eomt_iou
- eval_eomt_miou_temperature_scaling_cl
- eval_cityscapes_color
- eval_cityscapes_server
- eval_forwardTime


## eval_erfnet.py / eval_eomt.py

These scripts can be used to produce anomaly segmentation results (various anomaly metrics) on the validation datasets you can download here:

[validation datasets (Google Drive)](https://drive.google.com/file/d/1zcayoIIJztxKuHOIjmSjGoQBDy4RdETr/view)

**Examples:**

```bash
python eval_eomt.py \
  --loadDir ../trained_models \
  --loadWeights /path/to/your_checkpoint.ckpt \
  --input /path/to/Validation_Dataset/*/images/*.*
```

## eval_eomt_temperature_scaling_cl.py / eval_eomt_temperature_scaling_pl.py

These scripts run OOD evaluation with MSP temperature scaling and write results to:

- `eval/results/results_EoMT_Temperature_cl.txt`
- `eval/results/results_EoMT_Temperature_pl.txt`

```bash
python eval_eomt_temperature_scaling_cl.py \
  --loadDir ../trained_models \
  --loadWeights /path/to/your_checkpoint.ckpt \
  --input /path/to/Validation_Dataset/*/images/*.* \

python eval_eomt_temperature_scaling_pl.py \
  --loadDir ../trained_models \
  --loadWeights /path/to/your_checkpoint.ckpt \
  --input /path/to/Validation_Dataset/*/images/*.* \
```

## Cityscapes utilities

This code can be used to produce segmentation of the Cityscapes images in color for visualization purposes. By default it saves images in eval/save_color/ folder. You can also visualize results in visdom with --visualize flag.

- [**The Cityscapes dataset**](https://www.cityscapes-dataset.com/): Download the "leftImg8bit" for the RGB images and the "gtFine" for the labels. **Please note that for training you should use the "_labelTrainIds" and not the "_labelIds", you can download the [cityscapes scripts](https://github.com/mcordts/cityscapesScripts) and use the [conversor](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py) to generate trainIds from labelIds**

## eval_cityscapes_color.py

**Options:** Specify the Cityscapes folder path with '--datadir' option. Select the cityscapes subset with '--subset' ('val', 'test', 'train' or 'demoSequence'). For other options check the bottom side of the file.

**Examples:**

```bash
python eval_cityscapes_color.py --datadir /home/datasets/cityscapes/ --subset val
```

## eval_cityscapes_server.py

This code can be used to produce segmentation of the Cityscapes images and convert the output indices to the original 'labelIds' so it can be evaluated using the scripts from Cityscapes dataset (evalPixelLevelSemanticLabeling.py) or uploaded to Cityscapes test server. By default it saves images in eval/save_results/ folder.

**Options:** Specify the Cityscapes folder path with '--datadir' option. Select the cityscapes subset with '--subset' ('val', 'test', 'train' or 'demoSequence'). For other options check the bottom side of the file.

**Examples:**

```bash
python eval_cityscapes_server.py --datadir /home/datasets/cityscapes/ --subset val
```

## eval_erfnet_iou.py / eval_eomt_iou.py

These scripts can be used to calculate the IoU (mean and per-class) on Cityscapes val/train sets.

**Options:** Specify the Cityscapes folder path with '--datadir' option. Select the cityscapes subset with '--subset' ('val' or 'train'). For other options check the bottom side of the file.

**Examples:**

```bash
python eval_eomt_iou.py \
  --datadir "/path/to/Cityscapes" \
  --loadDir ../trained_models \
  --loadWeights epoch_106-step_19902_eomt.ckpt
```

## eval_eomt_miou_temperature_scaling_cl.py

This script runs a temperature scaling(cl) and writes mIoU results to `eval/results/results_EoMT_Temperature_mIoU_cl.txt`.

```bash
python eval_eomt_miou_temperature_scaling_cl.py \
  --datadir "/path/to/Cityscapes" \
  --loadDir ../trained_models \
  --loadWeights /path/to/your_checkpoint.ckpt \
```

## eval_forwardTime.py

This function loads a model specified by '-m' and enters a loop to continuously estimate forward pass time (fwt) in the specified resolution.

**Options:** Option '--width' specifies the width (default: 1024). Option '--height' specifies the height (default: 512). For other options check the bottom side of the file.

**Examples:**

```bash
python eval_forwardTime.py
```

**NOTE**: The pytorch code is a bit faster, but cudahalf (FP16) seems to give problems at the moment for some pytorch versions so this code only runs at FP32 (a bit slower).



