# Acute-stroke_Detection_Segmentation (ADS)

## Introduction

We provide a tool for detection and segmentation of ischemic acute and sub-acute strokes in brain diffusion weighted MRIs (DWIs). The deep learning networks were trained and tested on a large dataset of 2,348 clinical images, and further tested on 280 images of an external dataset. Our proposed model outperformed generic nets and patch-wise approaches, particularly in small lesions, with lower false positive rate, balanced precision and sensitivity, and robustness to data perturbs (e.g., artefacts, low resolution, technical heterogeneity). The agreement with human delineation rivaled the inter-evaluator agreement; the automated lesion quantification (e.g., volume) had virtually total agreement with human quantification. The method has minimal computational requirements and is fast: the lesion inference takes 20~30 seconds in CPU and the total processing, including registration and generation of results/report takes ~ 2.5 mins. The outputs, provided with a single command line, are: the predicted lesion mask, the lesion mask and the image inputs (DWI, B0, ADC) in standard space, MNI, and the quantification of the lesion per brain structure and per vascular territory.

<p align="middle">
    <img src="assets/Select2.gif", width="800" height="250">
</p>
<p align="middle">
    <img src="assets/Subject01_slices.png", width="410" height="240">
    <img src="assets/Subject02_slices.png", width="410" height="240">
</p>


## Directory
### Root
The `${ROOT}` is described as below.

```
${ROOT}
|-- data
    |-- Trained_Nets
    |-- examples
        |-- Subject01
        |-- Subject01
    |-- template
|-- codes
```

* `data` contains data like templates, image examples, and trained networks.
* `codes` contains ADS pipeline bin codes and the main function.

## Installation and Requirements
### Required Dependencies 

* python (version 3.7.7): Please make sure the version is at least 3.6+
* tensorFlow (version 2.0.0): The Deep Learning networks library for backend.
* h5py (version 2.10.0) : The networks model storage library. (please make sure its version is 2.10.0. Any different version might generate unexpected error when loading our pretrained models.) 
* niBabel (version 3.2.1): For loading NIFTI files.
* numpy (version 1.19.5): Gerenal computing array processing library.
* scipy (version 1.4.1): For image operation/processing. 
* dipy (version 1.4.0): For image registration
* scikit-image (version 0.18.1): For image operation/processing. 
* scikit-learn (version 0.24.1): Not necessary, but recommended because we will update codes with this dependency.

### STEP 1: Download ADS from github or google drive

* #### From github: 
    Cloned the codes (for unix system, similar steps should be sufficient for Windows) with :
    ```
    git clone https://github.com/Chin-Fu-Liu/Acute-stroke_Detection_Segmentation/
    ```
    Download pre-trained networks :
    1. Download pre-trained networks from [here (google drive) under uploading](https://drive.google.com/drive/)
    2. Unzip and put all the models ('MODEL_NAMES.h5' files) directly under `Trained_Nets` folder that is under your `Acute-stroke_Detection_Segmentation` main folder.
    
* #### From google drive: 
    If you are not familiar with github, you can just download the whole ADS package (ADS.zip file) from google drive [here (google drive) under uploading](https://drive.google.com/drive/) and unzip it to create the `Acute-stroke_Detection_Segmentation` main folder locally.

### STEP 2: Create virtual environment and activate the virtual environment:

We highly recommend creating a virtual enviroment for using this software. 

From a bash shell, create a virtual environment in a folder (FOLDER_FOR_ENVS/ENVS_FOLDER_NAME) that you want.

FOLDER_FOR_ENVS can be the path to the folder (`Acute-stroke_Detection_Segmentation`) you create and clone from github or google drive.
ENVS_FOLDER_NAME can be any name you like, ex: `ADS_ENV`.

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
$ pip install h5py==2.10.0
```
(We don't support GPU in this version yet, but it will be included in the next updated version. Hence, please also install `tensorflow-gpu`.)


### STEP 4: How to get the predicted stroke mask

Navigate to to the `/Acute-stroke_Detection_Segmentation/codes` folder, by `cd PATH_TO_/Acute_Stroke_Detection/codes/`

Under the `Acute-stroke_Detection_Segmentation/codes` folder, run 

```
python ADSRun.py -input PATH_TO/SUBJECTID_FOLDER 

```

#### The input format under `SUBJECTID_FOLDER`

The input images must be in Nifti format (.nii or .nii.gz). You can convert to Nifti using any available software/script (e.g., dcm2nii, MRICron, ImageJ).

`SUBJECTID_FOLDER` should be named by its SubjectID (see our example under "data/examples"). The data storage structure and naming must be as follows:

    |-- SUBJECTID_FOLDER
        |-- SUBJECTID_DWI.nii.gz
        |-- SUBJECTID_b0.nii.gz
        |-- SUBJECTID_ADC.nii.gz (optional)


The mandatory inputs are DWI and B0; ADC is optional. If no ADC is provided, it will be calculated with the b-value provided by the user (see option `-bvalue`). If no b-value is provided, it will use the default, b=1000. 


##### Note: The naming of input images is case sensitive.



### Options for ADSRun.py

For detail description, run -h for help as following
```
python ADSRun.py -h
```
`-input` is the path for  `SUBJECTID_FOLDER`

`-model` is the model name for segmenting lesions. It can be `DAGMNet_CH3`, `DAGMNet_CH2`, `UNet_CH3`, `UNet_CH2`, `FCN_CH3`, and `FCN_CH2`. These models were pretrianed by our data, as reported in our paper [cite:] `(defaut: DAGMNet_CH3)`

`-bvalue` is used to specify the b-value to calculate ADC, if ADC is not given.  If ADC is given under `SUBJECTID_FOLDER`, this option will be ignored. `(defaut: 1000)`

`-save_MNI`  is used to specify whether to save images in MNI space (DWI, b0, ADC, Normalized DWI and lesion predict). You can turn it off as `-save_MNI False`. `(defaut: True)`

`-generate_brainmask` is used to specify whether to generate brain mask in raw space, brain mask in MNI will be generated if `-save_MNI` is True. You can turn it on as `-save_MNI True`. `(defaut: False)`

`-generate_report`  is used to specify whether to generate the “lesion report”. The lesion report shows the total lesion volume and the total intracranial volume in the subject's original space, as well as the estimated number of lesion voxels per brain structure and per vascular territory, in MNI space. The folder "data/template" contains the eletronic atlases defining these areas as well a list of their volumes, in the MNI template. Therefore, users can calculate the relative distribution of the lesions (e.g., the percentage of a given vascular territory affected by the stroke). Be aware that these values are calculated by linear mapping to MNI space, therefore the precision highly depends on the particular brain morphology. You can turn it off as `-generate_report False`. `(defaut: True)`

`-generate_result_png` is used to specify whether to generate a figure (.png) of DWI, b0, ADC, and DWI aligned with lesion predict (blue contour) in the original image space. This figure is useful for immediate quality checking. You can turn it off as `-generate_result_png False`. `(defaut: True)`

For example, if you want to get a lesion predict on Subject01 with DAGMNet_CH3 model, as well as the images in MNI space, the lesion report, and the figure for quality control, you simply run the following line in your virtual environment under the `Acute_Stroke_Detection/codes` folder.

```
python ADSRun.py -input PATH_TO_Subject01_FOLDER
```

if you want to get a lesion predict on Subject01 with UNet_CH2 model, but not generating images in MNI and lesion report, you can run the following code in your virtual environment under the `Acute_Stroke_Detection/codes` folder.

```
python ADSRun.py -input PATH_TO_Subject01_FOLDER 
                 -model UNet_CH2
                 -save_MNI False
                 -generate_report False
                 -generate_result_png True
```

If you are not running the code under under the `Acute_Stroke_Detection/codes` folder, then you need to specify the path to `Acute_Stroke_Detection/codes` folder, as follows:
```
python PATHTO/Acute_Stroke_Detection/codes/ADSRun.py -input PATH_TO_Subject01_FOLDER
```


## News
* 2021.04.22. ADSv0.0 was uploaded.


## Reference  

## License 
This work is licensed under GNU General Public License v3.0, as found in the LICENSE file.

