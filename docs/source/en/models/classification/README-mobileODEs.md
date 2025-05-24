# Training MobileODEs on the ImageNet-A dataset

Single node 4-GPU training of `MobileODEv1` can be done using below command:

``` 
export CFG_FILE="config/classification/imagenet_a/mobilenetv1_ode.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc classification_results
```

***Note***: Do not forget to change the training and validation dataset locations in configuration files.

<details>
<summary>
Single node 8-GPU training of `MobileODEv2`
</summary>

``` 
export CFG_FILE="config/classification/imagenet_a/mobilenetv2_ode.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc classification_results
```
</details>


<details>
<summary>
Single node 8-GPU training of `MobileODEv1 + ViT`
</summary>

``` 
export CFG_FILE="config/classification/imagenet_a/mobilevitv1_ode.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc classification_results
```
</details>

<details>
<summary>
Single node 8-GPU training of `MobileODEv2 + ViT`
</summary>

``` 
export CFG_FILE="config/classification/imagenet_a/mobilevitv2_ode.yaml"
cvnets-train --common.config-file $CFG_FILE --common.results-loc classification_results
```
</details>
