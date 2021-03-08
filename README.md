# Acute_Stroke_Segmentation


## Introduction

In this study, we developed DL network ensembles for the detection and segmentation of acute and early subacute strokes in brain MRIs. Our network ensembles were trained and validated in 1390 clinical 3D images, and tested in 459 clinical 3D images, and evaluated on both MNI space and subject original raw space. To our best knowledge, this is the largest clinical set ever used. It is relatively much larger than the competitive study. In addition, we also evaluated our models on extra 499 not visible cases for false positive analysis. The results show our model ensembles are comparable to DeepMedic with comparable Dice score and precision but much lower false positive rate. We directly evaluated our models on an external dataset STIR (140 subjects for two image scan time points), which are clinical acute stroke subjects. The results show our model ensembles are generalized and robust on the external dataset. It sufficiently indicates our model has high potential in acute stroke detection application in clinical low-resolution images. The method fills all the requirements of speed, efficiency, robustness to data perturbs (e.g., imaging artifacts, low resolution, heterogeneity), and accessibility, for automatic acute stroke detection and segmentation.



### Root
The `${ROOT}` is described as below.
```
${ROOT}
|-- sample data
|-- codes
|-- trained_networks
|-- examples
|-- tutorial
```

## Reference  

## License 

This work is licensed under ...
