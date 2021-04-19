# Acute-stroke_Detection_Segmentation (ADS)

## Introduction

We provide a tool for detection and segmentation of ischemic acute and sub-acute strokes in brain diffusion weighted MRIs (DWIs). The deep learning networks were trained and tested on a large dataset of 2,348 clinical images, and further tested on 280 images of an external dataset. Our proposed model outperformed generic nets and patch-wise approaches, particularly in small lesions, with lower false positive rate, balanced precision and sensitivity, and robustness to data perturbs (e.g., artefacts, low resolution, technical heterogeneity). The agreement with human delineation rivaled the inter-evaluator agreement; the automated lesion quantification (e.g., volume) had virtually total agreement with human quantification. The method has minimal computational requirements, the lesion inference is fast (<1min in CPU) and provided with a single command line. We output the predicted lesion mask in the original space and in standard space, MNI (in addition to the inputs: DWI, ADC, B0) as well as the quantification of the lesion per brain structure and per vascular territory.

<p align="middle">
    <img src="assets/Select2.gif", width="1000" height="180">
</p>
<p align="middle">
    <img src="assets/Subject01_slices.png", width="480" height="300">
    <img src="assets/Subject02_slices.png", width="480" height="300">
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
* scikit-learn (version 0.24.1): Not necessary, but recommended because we will update codes with this dependency.

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
ENVS_FOLDER_NAME can be any name you like, ex: `ASD_ENV`.

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

(Use `pip install --user ` for local user)
```
$ pip install numpy nibabel scipy scikit-image scikit-learn
$ pip install dipy==1.4.0
$ pip install tensorflow==2.0.0 
$ pip install tensorflow-gpu==2.0.0
```
(We don't support GPU in this version, but it will be included in the next updated version. Hence, please also install `tensorflow-gpu`.)


### STEP 4: How to get the predicted stroke mask

Navigate to to the `/Acute_Stroke_Detection/codes` folder, by `cd PATH_TO_/Acute_Stroke_Detection/codes/`

Under the `Acute_Stroke_Detection/codes` folder, run 

```
python ASDRun.py -input SUBJECTID_FOLDER 
                 -model DAGMNet_CH3
```

#### The input format under `SUBJECTID_FOLDER` folder

The input format is Nifti (.nii or .nii.gz). The user can convert to this format using any software / script (e.g., dcm2nii, MRICron, ImageJ).

`SUBJECTID_FOLDER` should be named by its SubjectID, as in our example folder, we put it like `Subject01` or `Subject02`.
Each `SUBJECTID_FOLDER` folder should at least contain DWI and b0 images. And the data storage structure and naming format should be as following:

    |-- SUBJECTID_FOLDER
        |-- SUBJECTID_DWI.nii.gz
        |-- SUBJECTID_b0.nii.gz
        |-- SUBJECTID_ADC.nii.gz (optional)


The mandatory inputs are the DWI and the B0 MRIs. The ADC is optional. 

If no ADC is provided, it will be calculated with the b-value provided by the user. 

If no value is provided, the default b=1000 will be used to calculate ADC with given DWI and b0 MRIs. 

The naming is case sensitive.




### Options for ASDRun.py

For detail description, run -h for help as following
```
python ASDRun.py -h
```
`-input ` is the path for  `SUBJECTID_FOLDER`

`-model ` is the model name for segmenting lesions. It can be `DAGMNet_CH3`, `DAGMNet_CH2`, `UNet_CH3`, `UNet_CH2`, `FCN_CH3`, and `FCN_CH2`. They are pretrianed model by our data and specified in our paper [cite:]

`-save_MNI` is used to specify whether to save images in MNI (DWI, b0, ADC, Normalized DWI and, lesion predict). It's True by default.  you can turn it off as `-save_MNI False`

`-generate_report` is used to specify whether to generate lesion report. lesion report shows estimated lesion volume in total and in each vascular region or lobe area. It's True by default.  you can turn it off as `-generate_report False`

`-generate_result_png` is used to specify whether to generate the lesion predict result in a png file. In the png file, the order of columns are DWI, b0, ADC, and DWI aligned with lesion predict (blue contour) in the original image space. It's True by default.  you can turn it off as `-generate_result_png False`

For example, if you want to get a lesion predict on Subject01 with UNet_CH2 model, but not generating images in MNI and lesion report. You can run the following code in your virtual environment under the `Acute_Stroke_Detection/codes` folder.

```
python ASDRun.py -input PATH_to_Subject01_FOLDER 
                 -model UNet_CH2
                 -save_MNI False
                 -generate_report False
                 -generate_result_png True
```



## News
* 2021.04.16. examples are updated. 


## Reference  

## License 
This work is licensed under GNU General Public License v3.0, as found in the LICENSE file.

