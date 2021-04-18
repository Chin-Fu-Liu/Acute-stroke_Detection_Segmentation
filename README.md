# Acute_Stroke_Detection

## Introduction

In this study, we developed DL network ensembles for the detection and segmentation of acute and early subacute strokes in brain MRIs. Our network ensembles were trained and validated in 1390 clinical 3D images, and tested in 459 clinical 3D images, and evaluated on both MNI space and subject original raw space. To our best knowledge, this is the largest clinical set ever used. It is relatively much larger than the competitive study. In addition, we also evaluated our models on extra 499 not visible cases for false positive analysis. The results show our model ensembles are comparable to DeepMedic with comparable Dice score and precision but much lower false positive rate. We directly evaluated our models on an external dataset STIR (140 subjects for two image scan time points), which are clinical acute stroke subjects. The results show our model ensembles are generalized and robust on the external dataset. It sufficiently indicates our model has high potential in acute stroke detection application in clinical low-resolution images. The method fills all the requirements of speed, efficiency, robustness to data perturbs (e.g., imaging artifacts, low resolution, heterogeneity), and accessibility, for automatic acute stroke detection and segmentation.

<p align="middle">
    <img src="assets/exp_pic1.PNG", width="600" height="180">
</p>
<p align="middle">
    <img src="assets/exp_pic2.PNG", width="600" height="360">
</p>


## Directory
### Root
The `${ROOT}` is described as below.

```
${ROOT}
|-- data
    |-- Trained_Nets
    |-- examples
    |-- template
|-- codes
|-- tutorial
```

    * `data` contains data like templates, image examples, and trained networks.
    * `codes` contains ASD pipeline bin codes and the main function.

## Installation and Requirements
### Required Dependencies 

    * python (version 3.7.7): Please make sure the version is at least 3.6+
    * tensorFlow (version 2.0.0): The Deep Learning networks library for backend.
    * niBabel (version 3.2.1): For loading NIFTI files.
    * numpy (version 1.19.5): Gerenal computing array processing library.
    * scipy (version 1.4.1): For image operation/processing. 
    * dipy (version 1.4.0): For image registration
    * scikit-image (version 0.18.1): For image operation/processing. 
    * scikit-learn (version 0.24.1): Not necessary, but recommended to install as well, because we will update codes which has this dependency in the future. 

### STEP 1: Download ASD from github or google drive

Cloned the codes (for unix system, similar steps should be sufficient for Windows) with :
```
git clone https://github.com/Chin-Fu-Liu/Acute_Stroke_Detection/
```
If you are not familiar with github, you can just download the whole ASD package (ASD.zip file) from google drive [here (google drive) under uploading](https://drive.google.com/drive/) and unzip it to create the `Acute_Stroke_Detection` main folder locally.

### Download pre-trained networks :
1. Download pre-trained networks from [here (google drive) under uploading](https://drive.google.com/drive/)
2. Unzip and put all the models ('MODEL_NAMES.h5' files) directly under `Trained_Nets` folder that is under your `Acute_Stroke_Detection` main folder.


### STEP 2: Create virtual environment and activate the virtual environment:

We highly recommend creating a virtual enviroment for using this software. 

From a bash shell, create a virtual environment in a folder (FOLDER_FOR_ENVS/ENVS_FOLDER_NAME) that you want.

FOLDER_FOR_ENVS can be the path to the folder (`Acute_Stroke_Detection`) you create and clone from github or google drive.
ENVS_FOLDER_NAME can be any name you like, like `ASD_ENV`

Using Conda:
```
conda create -p FOLDER_FOR_ENVS/ENVS_FOLDER_NAME python=3.6.5 -y
source activate FOLDER_FOR_ENVS/ENVS_FOLDER_NAME
```
Using Virtualenv:
```
virtualenv -p python3 FOLDER_FOR_ENVS/ENVS_FOLDER_NAME     # Use python up to 3.6+
source FOLDER_FOR_ENVS/ENVS_FOLDER_NAME/bin/activate      
```

###  STEP 3: Install all Dependencies as follows
```
$ pip install numpy nibabel scipy scikit-image scikit-learn
$ pip install dipy==1.4.0
$ pip install tensorflow-gpu==2.0.0
```


### STEP 4: How to get new predict

Navigate to to the `/Acute_Stroke_Detection/codes` folder

In the `Acute_Stroke_Detection/codes` folder, run 

```bash
python ASDRun.py -input $SUBJECT_FOLDER 
                 -model DAGMNet_CH3
```



## News
* 2021.04.16. examples are updated. 


## Reference  

## License 
This work is licensed under GNU General Public License v3.0, as found in the LICENSE file.

