# Object detection using SSDLite on MS-COCO

## Training detection network on the MS-COCO dataset

Single node training of `SSDLite-MobileODE` with single 4090 GPU.

``` 
  PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/detection/BUSI_coco/mobilenetv1_ode.yaml --common.results-loc ssdlite_mobilenetv1_ode_results --common.override-kwargs model.classification.pretrained="LOCATION_OF_CLASSIFICATION_CHECKPOINT"
```
