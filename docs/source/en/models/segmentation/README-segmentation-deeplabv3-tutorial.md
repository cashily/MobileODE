# Semantic Segmentation using DeepLabv3

## Training segmentation network

Single node training of DeepLabv3 with any classification backbone, can be done using the below command:

``` 
  export CONFIG_FILE="PATH_TO_CONFIG_FILE"
  export IMAGENET_PRETRAINED_WTS="LOCATION_OF_IMAGENET_WEIGHTS"
  PYTHONWARNINGS="ignore" cvnets-train --common.config-file $CONFIG_FILE --common.results-loc deeplabv3_results --model.classification.pretrained $IMAGENET_PRETRAINED_WTS
```


***Note***: Do not forget to change the training and validation dataset locations in configuration files.

## Quantitative evaluation

Mean intersection over union (mIoU) score can be computed on segmentation dataset using the below command:

```
 export CFG_FILE="PATH_TO_CONFIG_FILE"
 export DEEPLABV3_MODEL_WEIGHTS="PATH_TO_MODEL_WEIGHTS" 
 CUDA_VISIBLE_DEVICES=0 cvnets-eval-seg --common.config-file $CFG_FILE --common.results-loc deeplabv3_results --model.segmentation.pretrained $DEEPLABV3_MODEL_WEIGHTS --evaluation.segmentation.resize-input-images --evaluation.segmentation.mode validation_set
 ```

## Qualitative evaluation

An example command to run segmentation on an image using `DeepLabv3-MobileODEv2` model trained on the PASCAL VOC 2012 is given below
``` 
 export IMG_PATH="LOCATION_OF_IMAGE"
 export CFG_FILE="PATH_TO_CONFIG_FILE"
 export MODEL_WEIGHTS="PATH_TO_MODEL_WEIGHTS"
 cvnets-eval-seg --common.config-file $CFG_FILE --common.results-loc deeplabv3_results --model.segmentation.pretrained $MODEL_WEIGHTS --model.segmentation.n-classes 21 --evaluation.segmentation.resize-input-images --evaluation.segmentation.mode single_image --evaluation.segmentation.path "${IMG_PATH}" \
 --evaluation.segmentation.save-masks \
 --evaluation.segmentation.apply-color-map \
 --evaluation.segmentation.save-overlay-rgb-pred
```