#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

try:
    import tensorflow.keras
#     print ('load keras from tensorflow package')
except:
    print ('update your tensorflow')
import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.compat.v2.keras.utils import multi_gpu_model

# # Use on GPU
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = "" # or '1' or whichever GPU is available on your machine

# for save model
import h5py

# default
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import scipy.io as sio
import nibabel as nib
import scipy
import time

from skimage import measure
from matplotlib import gridspec

from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.data.fetcher import fetch_syn_data, read_syn_data
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)
from dipy.align._public import SSDMetric
from dipy.align import (affine_registration, center_of_mass, translation,
                        rigid, affine, register_dwi_to_template)
from dipy.segment.mask import median_otsu
from dipy.core.histeq import histeq

from scipy.ndimage import morphology, gaussian_filter
from scipy.special import erf
from scipy.optimize import minimize,leastsq, curve_fit

# from pylab import *

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.warn("deprecated", DeprecationWarning)

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    
def get_new_NibImgJ(new_img, temp_imgJ, dataType=np.float32):
    temp_imgJ.set_data_dtype(dataType)
    img_header = temp_imgJ.header
    img_header['glmax'] = np.max(new_img)
    img_header['glmin'] = np.min(new_img)
    new_imgJ = nib.Nifti1Image(new_img,temp_imgJ.affine,img_header)
    return new_imgJ

def load_img_AffMat(img_fnamePth):
    imgJ = nib.load(img_fnamePth)
    img = np.squeeze(imgJ.get_fdata())
    img_AffMat = imgJ.affine
    return imgJ, img, img_AffMat

def MNIdePadding_imgJ(imgJ):
    img = np.squeeze(imgJ.get_fdata())
    de_img = img[5:186, 3:220, 5:186]
    
    deimg_AffMatt = imgJ.affine
    deimg_AffMatt[0,3] = 91
    deimg_AffMatt[1,3] = -109
    deimg_AffMatt[2,3] = -91
    
    img_header = imgJ.header
    img_header['glmax'] = np.max(de_img)
    img_header['glmin'] = np.min(de_img)
    
    de_imgJ = nib.Nifti1Image(de_img, deimg_AffMatt, img_header)
    return de_imgJ

def MNIPadding_imgJ(imgJ):
    img = np.squeeze(imgJ.get_fdata())
    pad_img = np.zeros((192, 224, 192))
    pad_img[5:186, 3:220, 5:186] = img
    
    padimg_AffMatt = imgJ.affine
    padimg_AffMatt[0,3] = 96
    padimg_AffMatt[1,3] = -112
    padimg_AffMatt[2,3] = -96
    
    img_header = imgJ.header
    img_header['glmax'] = np.max(pad_img)
    img_header['glmin'] = np.min(pad_img)
    
    pad_imgJ = nib.Nifti1Image(pad_img, padimg_AffMatt, img_header)
    return pad_imgJ

def Cal_ADC(Dwi_ImgJ, B0_ImgJ, ADCPath, bvalue=1000):
    Dwi_img = np.squeeze(Dwi_ImgJ.get_fdata())
    B0_img = np.squeeze(B0_ImgJ.get_fdata())
    e = 1e-10;
    ADC_img =  (-np.log((Dwi_img+e)/(B0_img+e))/bvalue)*(B0_img>0)*(Dwi_img>0)
    ADC_ImgJ = get_new_NibImgJ(ADC_img, Dwi_ImgJ, dataType=np.float32)
    nib.save(ADC_ImgJ, ADCPath)
 
   
def Sequential_Registration_b0(static, static_grid2world, moving, moving_grid2world, level_iters = [5], sigmas = [3.0], factors = [2]):
    pipeline = [center_of_mass, translation, rigid, affine]
    xformed_img, reg_affine = affine_registration(
        moving,
        static,
        moving_affine=moving_grid2world,
        static_affine=static_grid2world,
        nbins=16,
        metric='MI',
        pipeline=pipeline,
        level_iters=level_iters,
        sigmas=sigmas,
        factors=factors)
    affine_map = AffineMap(reg_affine,
                       static.shape, static_grid2world,
                       moving.shape, moving_grid2world)
    return xformed_img, reg_affine, affine_map

def Stroke_closing(img):
    new_img = np.zeros_like(img)
    new_img = morphology.binary_closing(img, structure=np.ones((2,2,2)))
    return new_img

def Stroke_connected(img, connect_radius=1):
    return morphology.binary_dilation(img, morphology.ball(radius=connect_radius))


def remove_small_objects(img, remove_max_size=5):
    binary = np.zeros_like(img)
    binary[binary>0] = 1
    labels = morphology.label(binary)
    labels_num = [len(labels[labels==each]) for each in np.unique(labels)]
#     print(np.unique(labels))
#     print(labels_num)
    new_img = copy.copy(img)
    for index in np.unique(labels):
        if labels_num[index]<remove_max_size:
            new_img[labels==index] = 0
    return new_img

def check_regions(img):
    binary = np.zeros_like(img)
    binary[binary>0] = 1
    labels = morphology.label(binary)
    labels_num = [len(labels[labels==each]) for each in np.unique(labels)]
    print(np.unique(labels))
    print(labels_num)
    
def remove_small_objects(img, remove_max_size=5, structure = np.ones((3,3))):
    binary = img
    binary[binary>0] = 1
    labels = np.array(scipy.ndimage.label(binary, structure=structure))[0]
    labels_num = [len(labels[labels==each]) for each in np.unique(labels)]
    new_img = img
    for index in np.unique(labels):
        if labels_num[index]<remove_max_size:
            new_img[labels==index] = 0
    return new_img

def remove_small_objects_InSlice(img, remove_max_size=5,  structure = np.ones((3,3))):
    img = np.squeeze(img)
    new_img = np.zeros_like(img)
    
    for idx in range(img.shape[-1]):
        new_img[:,:,idx] = remove_small_objects(img[:,:,idx],remove_max_size=remove_max_size, structure=structure)
    return new_img
    
def get_MaskNet_MNI(model, Dwi_MNI_img, B0_MNI_img):

    dwi = Dwi_MNI_img[0::4,0::4,0::4,np.newaxis] # Down sample for MaskNet, dim should be [48, 56, 48, 1]
    dwi  = (dwi-np.mean(dwi))/np.std(dwi)

    b0 = B0_MNI_img[0::4,0::4,0::4, np.newaxis] # Down sample for MaskNet, dim should be [48, 56, 48, 1]
    b0  = (b0-np.mean(b0))/np.std(b0)

    x = np.expand_dims(np.concatenate((dwi,b0),axis=3), axis=0)

    y_pred = model.predict(x, verbose=0)
    y_pred = (np.squeeze(y_pred)>0.5)*1.0

    dilate_mask = y_pred
    mask_label, num_features = scipy.ndimage.measurements.label(dilate_mask)
    dilate_mask = (mask_label == mask_label[24,28,24])*1
    # for i in range(48):
    #     dilate_mask[:,:,i] = scipy.ndimage.binary_dilation(dilate_mask[:,:,i])*1.0
    dilate_mask = Stroke_closing(dilate_mask)
    dilate_mask = morphology.binary_fill_holes(dilate_mask)

    upsampling_mask = np.repeat(np.repeat(np.repeat(dilate_mask, 4, axis=0), 4, axis=1), 4, axis=2)

    return upsampling_mask

def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def qfunc(x):
    return 0.5-0.5*scipy.special.erf(x/np.sqrt(2))

def get_dwi_normalized(Dwi_ss_MNI_img,mask_raw_MNI_img):
    Dwi_d = Dwi_ss_MNI_img[mask_raw_MNI_img>0.5]

    md = scipy.stats.mode(Dwi_d.astype('int16'))[0][0]
    if md > np.mean(Dwi_d):
        p0_mu = md
    else:
        p0_mu = np.mean(Dwi_d)
        
    Dwi_hist, xData = np.histogram(Dwi_d, bins=np.arange(np.max(Dwi_d)),  density=True)
    xData=(xData[1:]+xData[:-1])/2 # for len(x)==len(y)
    
    bounds = ([0, 0, -np.inf, 0, 0, -np.inf], [np.inf, np.inf, np.inf, p0_mu, np.inf, np.inf])
    params,cov=curve_fit(bimodal,xData,Dwi_hist, bounds=bounds, p0=(p0_mu,1,1, 0,1,1 ))
    
    mu1 = params[0]
    sigma1 = params[1]
    a1 = params[2]

#     mu2 = params[3]
#     sigma2 = params[4]
#     a2 = params[5]
    
    Dwi_ss_MNI_norm_img = (Dwi_ss_MNI_img-mu1)/sigma1
    return Dwi_ss_MNI_norm_img

def get_Prob_IS(Dwi_ss_MNI_norm_img, ADC_ss_MNI_img, mask_raw_MNI_img, 
                TemplateDir, 
                model_vars = [2,1.5,4,0.5,2,2]):
    template_fnamePth = os.path.join(TemplateDir, 'normal_mu_dwi_Res_ss_MNI_scaled_normalized.nii.gz')   
    _, normal_dwi_mu_img, _ = load_img_AffMat(template_fnamePth)

    template_fnamePth = os.path.join(TemplateDir, 'normal_std_dwi_Res_ss_MNI_scaled_normalized.nii.gz')   
    _, normal_dwistd_img, _ = load_img_AffMat(template_fnamePth)

    template_fnamePth = os.path.join(TemplateDir, 'normal_mu_ADC_Res_ss_MNI_normalized.nii.gz')   
    _, normal_adc_mu_img, _ = load_img_AffMat(template_fnamePth)

    template_fnamePth = os.path.join(TemplateDir, 'normal_std_ADC_Res_ss_MNI_normalized.nii.gz')   
    _, normal_adc_std_img, _ = load_img_AffMat(template_fnamePth)
    
    fwhm = model_vars[0];
    g_sigma = fwhm/2/np.sqrt(2*np.log(2));
    alpha_dwi = model_vars[1];
    lambda_dwi = model_vars[2];
    alpha_adc = model_vars[3];
    lambda_adc = model_vars[4];
    id_isch_zth = model_vars[5];

    img = (Dwi_ss_MNI_norm_img - np.mean(Dwi_ss_MNI_norm_img)) / np.std(Dwi_ss_MNI_norm_img)
    for i in range(img.shape[-1]):
        img[:,:,i] = gaussian_filter(img[:,:,i], g_sigma)
    dissimilarity = np.tanh((img - normal_dwi_mu_img)/normal_dwistd_img/alpha_dwi)
    dissimilarity[dissimilarity<0] = 0
    dissimilarity = dissimilarity ** lambda_dwi
    dissimilarity[Dwi_ss_MNI_norm_img<id_isch_zth] = 0
    dwi_H2 = dissimilarity*(mask_raw_MNI_img>0.5)*1.0

    img = (ADC_ss_MNI_img - np.mean(ADC_ss_MNI_img)) / np.std(ADC_ss_MNI_img)
    for i in range(img.shape[-1]):
        img[:,:,i] = gaussian_filter(img[:,:,i], g_sigma)
    dissimilarity = np.tanh((img - normal_adc_mu_img)/normal_adc_std_img/alpha_adc)
    dissimilarity[dissimilarity>0] = 0
    dissimilarity = (-dissimilarity) ** lambda_adc
    adc_H1 = dissimilarity*(mask_raw_MNI_img>0.5)*1.0


    id_isch = Dwi_ss_MNI_norm_img
    id_isch = (1-qfunc(id_isch/id_isch_zth))*(id_isch>id_isch_zth)

    Prob_IS = dwi_H2*adc_H1*id_isch*(mask_raw_MNI_img>0.5)*1.0
    
    return Prob_IS

def get_stroke_seg_MNI(model, dwi_img, adc_img, Prob_IS=None, N_channel=3, DS=2):
    stroke_pred_resampled =  np.squeeze(np.zeros_like(dwi_img))
    for x_idx, y_idx, slice_idx in [(x,y,z) for x in range(DS) for y in range(DS) for z in range(2*DS)]:
        if N_channel==3:
            dwi_DS_img = dwi_img[x_idx::DS,y_idx::DS,slice_idx::2*DS, np.newaxis]
            adc_DS_img = adc_img[x_idx::DS,y_idx::DS,slice_idx::2*DS, np.newaxis]
            Prob_IS_DS_img = Prob_IS[x_idx::DS,y_idx::DS,slice_idx::2*DS, np.newaxis]
            imgs_input = np.expand_dims(np.concatenate((dwi_DS_img,adc_DS_img,Prob_IS_DS_img),axis=3), axis=0)
        elif N_channel==2:
            dwi_DS_img = dwi_img[x_idx::DS,y_idx::DS,slice_idx::2*DS, np.newaxis]
            adc_DS_img = adc_img[x_idx::DS,y_idx::DS,slice_idx::2*DS, np.newaxis]
            imgs_input = np.expand_dims(np.concatenate((dwi_DS_img,adc_DS_img),axis=3), axis=0)
        stroke_pred = model.predict(imgs_input, verbose=0)[0]
        stroke_pred = np.squeeze(stroke_pred)
        stroke_pred_resampled[x_idx::DS,y_idx::DS,slice_idx::2*DS] = stroke_pred
    
    stroke_pred_tmp = (stroke_pred_resampled>0.5)
#     stroke_pred_tmp = remove_small_objects_InSlice(stroke_pred_tmp)
    stroke_pred_tmp = Stroke_closing(stroke_pred_tmp)
    stroke_pred_tmp = morphology.binary_fill_holes(stroke_pred_tmp)
    
    return stroke_pred_tmp

def get_DirPaths():
    CodesDir = os.path.dirname(os.path.abspath(__file__)) + '/'
#     print(CodesDir)
    ProjectDir = os.path.join('/'.join(CodesDir.split('/')[0:-2]),'')
    TemplateDir = os.path.join(ProjectDir,'data','template','')
    TrainedNetsDir = os.path.join(ProjectDir,'data','Trained_Nets','')
    return CodesDir, ProjectDir, TemplateDir, TrainedNetsDir

def gen_result_png(SubjDir, 
                        lesion_name='Lesion_Predict',
                        wspace=-0.1,
                        hspace=-0.5
                       ):

    plt.rcParams["axes.grid"] = False
    
    SubjID = os.path.join(SubjDir,'').split('/')[-2]
    DwiPath = os.path.join(SubjDir, SubjID + '_DWI.nii.gz')
    B0Path = os.path.join(SubjDir, SubjID + '_b0.nii.gz')
    ADCPath = os.path.join(SubjDir, SubjID + '_ADC.nii.gz')
    LPPath = os.path.join(SubjDir, SubjID + '_' + lesion_name + '.nii.gz')
    
    Dwi_imgJ, Dwi_img, _ = load_img_AffMat(DwiPath)
    B0_imgJ, B0_img, B0_AffMat = load_img_AffMat(B0Path)
    ADC_imgJ, ADC_img, _ = load_img_AffMat(ADCPath)
    LP_imgJ, LP_img, _ = load_img_AffMat(LPPath)  
    

    # sli = Dwi_img.shape[2] // 2 
    fig = plt.figure(figsize=(12, 3*Dwi_img.shape[2]))
    gs0 = gridspec.GridSpec(nrows=Dwi_img.shape[2], ncols=4)

    plt.style.use('grayscale')
    for sli in range(Dwi_img.shape[2]):

        ax0 = plt.subplot(gs0[sli, 0])
        ax0.imshow(np.rot90(Dwi_img[:, :, sli]), cmap=plt.cm.gray)
        plt.axis('off')
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])

        ax0 = plt.subplot(gs0[sli, 1])
        ax0.imshow(np.rot90(B0_img[:, :, sli]), cmap=plt.cm.gray)
        plt.axis('off')
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])

        ax0 = plt.subplot(gs0[sli, 2])
        ax0.imshow(np.rot90(ADC_img[:, :, sli]), cmap=plt.cm.gray)
        plt.axis('off')
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])


        prc = measure.find_contours(np.rot90(LP_img[:,:,sli]), 0.8)
        ax0 = plt.subplot(gs0[sli, 3])
        ax0.imshow(np.rot90(Dwi_img[:, :, sli]), cmap=plt.cm.gray)
        for contour in prc:
            ax0.plot(contour[:, 1], contour[:, 0],  color='deepskyblue', linewidth=1, alpha = 1)
        plt.axis('off')
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])

    plt.tight_layout()
    gs0.update(wspace=wspace, hspace=hspace)
    plt.savefig(os.path.join(SubjDir, SubjID + '_' + lesion_name + '_result.png'), bbox_inches = "tight")
#     plt.show()
    plt.close()
    
def get_VasLobeTemp(TemplateDir):
    vas_pth = os.path.join(TemplateDir,'ArterialAtlas_padding.nii.gz')
    lobe_pth = os.path.join(TemplateDir,'lobe_atlas_padding.nii.gz') 

    vas_imgJ, vas_img, vas_AffMat = load_img_AffMat(vas_pth)
    lobe_imgJ, lobe_img, lobe_AffMat = load_img_AffMat(lobe_pth)
    
    vas_table_pth = os.path.join(TemplateDir,'ArterialAtlasLables.txt')
    with open(vas_table_pth) as f:
        vas_contents = f.readlines()
    vas_contents = vas_contents[7:] #drop file header
    
    lobe_table_pth = os.path.join(TemplateDir,'LobesLabelLookupTable.txt') 
    with open(lobe_table_pth) as f:
        lobe_contents = f.readlines()
    
    vas_L1 = [ _.split('\t')[2] for _ in vas_contents]
    vas_L2 = [ _.split('\t')[5] for _ in vas_contents]
    vas_L1_name = [ _.split('\t')[0] for _ in vas_contents]
    
    lobe_L1 = [ _.split('\t')[1].replace('\n','') for _ in lobe_contents]
    lobe_L1
    
    def get_L2_label_idx(L):
        if L == 'ACA':
            return 1
        elif L == 'MCA':
            return 2
        elif L == 'PCA':
            return 3
        elif L == 'VB':
            return 4
        else:
            return 0
    vas_L2 = ['ACA', 'MCA', 'PCA', 'VB']

    vas_combine = np.zeros_like(vas_img)
    for idx in range(len(vas_L2)):
        vas_combine[(vas_img==idx+1)] = get_L2_label_idx(vas_L2[idx])
        
#     print(np.max(vas_combine))
    
    return vas_img, vas_combine, lobe_img, vas_L1, vas_L1_name, vas_L2, lobe_L1

def get_category_features(stroke_img, temp):
    v=np.zeros(np.int(temp.max()))
    inters = (stroke_img>0.5) * temp
    for i in range(np.int(temp.max())):
        v[i] = np.sum(inters==(i+1))
    return v


def gen_lesion_report(SubjDir, SubjID, lesion_img, ICV_vol, Lesion_vol, TemplateDir):
    
    vas_img, vas_combine, lobe_img, vas_L1, vas_L1_name, vas_L2, lobe_L1 = get_VasLobeTemp(TemplateDir)
    
#     Lesion_vol = np.sum(lesion_img)
    vas_v_L1 = get_category_features(lesion_img, vas_img)
    vas_v_L2 = get_category_features(lesion_img, vas_combine)
    lobe_v_L1 = get_category_features(lesion_img, lobe_img)
    
#     print(np.max(vas_combine))
    ReportTxt_pth = os.path.join(SubjDir, SubjID + '_volume_brain_regions.txt')
    with open(ReportTxt_pth, 'w') as f:
        
        f.write('intracranial volume \t%d' % (ICV_vol) + '\n\n')
        f.write('stroke volume \t%d' % (Lesion_vol) + '\n\n')
        
        f.write('vascular territory\tnumber of voxel\n')
        
        for idx in range(len(vas_L1)):
            f.write(vas_L1[idx]+'\t%d\t'% vas_v_L1[idx] + vas_L1_name[idx] + '\n')
            
        f.write('\n')
        f.write('vascular territory 2\tnumber of voxel\n')
        for idx in range(len(vas_L2)):
            f.write(vas_L2[idx]+'\t%d\t'% vas_v_L2[idx] + '\n')
        
        f.write('\n')
        f.write('area\tnumber of voxel\n')
        for idx in range(len(lobe_L1)):
            f.write(lobe_L1[idx]+'\t%d\t'% lobe_v_L1[idx] + '\n')  
        
            
def ADS_pipeline(SubjDir, 
                 SubjID,
                 TemplateDir,
                 MaskNet_name,
                 Lesion_model_name,
                 N_channel,
                 level_iters = [3], 
                 sigmas = [3.0], 
                 factors = [2], 
                 lesion_name='Lesion_Predict',
                 bvalue = 1000,
                 save_MNI=True, 
                 generate_report=True,
                 generate_result_png=True
                ):
    start_time = time.time()

    # loading dwi, b0
    print('------ Loading DWI, b0 ------')
    
    # check nii or nii.gz
    DwiPath = os.path.join(SubjDir, SubjID + '_DWI.nii.gz')
    if not os.path.exists(DwiPath):
        DwiPath = os.path.join(SubjDir, SubjID + '_DWI.nii')
        
    B0Path = os.path.join(SubjDir, SubjID + '_b0.nii.gz')
    if not os.path.exists(B0Path):
        B0Path = os.path.join(SubjDir, SubjID + '_b0.nii')
        
    if not os.path.exists(DwiPath):
        sys.exit('No SujbectID_DWI.nii or SujbectID_DWI.nii.gz file under subject folder!') 
    else:
        Dwi_imgJ, Dwi_img, _ = load_img_AffMat(DwiPath)
        
        pixdim = Dwi_imgJ.header.get_zooms()
        VolSpacing = 1
        for _ in pixdim:
            VolSpacing = VolSpacing*_
        
    if not os.path.exists(B0Path):
        sys.exit('No SujbectID_b0.nii or SujbectID_b0.nii.gz file under subject folder!') 
    else:
        B0_imgJ, B0_img, B0_AffMat = load_img_AffMat(B0Path)

    # calculate ADC
    ADCPath = os.path.join(SubjDir, SubjID + '_ADC.nii')
    if not os.path.exists(ADCPath):
        ADCPath = os.path.join(SubjDir, SubjID + '_ADC.nii.gz')
        
    if not os.path.exists(ADCPath):
        print('------ Calculating ADC with b-value=%d ------' % bvalue)
        Cal_ADC(Dwi_imgJ, B0_imgJ, ADCPath, bvalue)
    
    # loading ADC
    print('------ Loading ADC ------')
    ADC_imgJ, ADC_img, _ = load_img_AffMat(ADCPath)
    
    # loading template
    print('------ Loading JHU_SS_b0_padding template------')
    JHU_B0_withskull_fnamePth = os.path.join(TemplateDir, 'JHU_SS_b0_padding.nii.gz')
    JHU_B0_imgJ, JHU_B0_img, JHU_B0_AffMat = load_img_AffMat(JHU_B0_withskull_fnamePth)
    
    # mapping to MNI with skull for MaskNet
    start_time = time.time()
    print('------ Mapping to MNI with skull for MaskNet------')
    B0_MNI_img, reg_affine, affine_map = Sequential_Registration_b0(static=JHU_B0_img, 
                                                                static_grid2world=JHU_B0_AffMat,
                                                                moving=B0_img,
                                                                moving_grid2world=B0_AffMat,
                                                                level_iters = level_iters, 
                                                                sigmas = sigmas, 
                                                                factors = factors
                                                               )
    print('------ Finished mapping to MNI with skull for MaskNet------') 
    print('It takes %.2f seconds'% (time.time() - start_time))
    start_time = time.time()
    
    Dwi_MNI_img = affine_map.transform(Dwi_img)
#     ADC_MNI_img = affine_map.transform(ADC_img)
    
    # Loading MaskNet
    print('------ Loading MaskNet------')
    MaskNet = load_model(MaskNet_name, compile=False)
    
    # get brain mask in raw space
    print('------ Inferencing brain mask------')
    mask_MNI_img = get_MaskNet_MNI(MaskNet, Dwi_MNI_img, B0_MNI_img)
    mask_raw_img = affine_map.transform_inverse((mask_MNI_img>0.5)*1, interpolation='nearest')
    
    print('------ Finished inferencing brain mask------')
    print('It takes %.2f seconds'% (time.time() - start_time))
    start_time = time.time()
    
    # get skull stripped dwi, b0, adc
    print('------ skull-stripping------')
    Dwi_ss_img = Dwi_img*mask_raw_img
    B0_ss_img = B0_img*mask_raw_img
    ADC_ss_img = ADC_img*mask_raw_img
    
    # loading template_ss 
    print('------ Loading JHU_SS_b0_ss_padding template------')
    JHU_B0_ss_fnamePth = os.path.join(TemplateDir, 'JHU_SS_b0_ss_padding.nii.gz')  
    JHU_B0_ss_imgJ, JHU_B0_ss_img, JHU_B0_ss_AffMat = load_img_AffMat(JHU_B0_ss_fnamePth)
    
    # get mapping to MNI without skull for lesion detection model
    print('------ Mapping to MNI without skull for lesion detection model ------')
    B0_ss_MNI_img, reg_affine, affine_map = Sequential_Registration_b0(static=JHU_B0_ss_img, 
                                                                    static_grid2world=JHU_B0_ss_AffMat,
                                                                    moving=B0_ss_img,
                                                                    moving_grid2world=B0_AffMat,
                                                                    level_iters = level_iters, 
                                                                    sigmas = sigmas, 
                                                                    factors = factors
                                                                   )
    # mapping images to MNI
    Dwi_ss_MNI_img = affine_map.transform(Dwi_ss_img)
    ADC_ss_MNI_img = affine_map.transform(ADC_ss_img)
    mask_raw_MNI_img = affine_map.transform(mask_raw_img, interpolation='nearest') 
    
    
        
    print('------ Mapping to MNI without skull for lesion detection model ------')
    print('It takes %.2f seconds'% (time.time() - start_time))
    start_time = time.time()
    
    # get normalized dwi
    print('------ Normalizing dwi------')
    Dwi_ss_MNI_norm_img = get_dwi_normalized(Dwi_ss_MNI_img,mask_raw_MNI_img)
    
    # get Prob. IS
    if N_channel==3:
        print('------ Calculating Prob. IS Map for CH3 ------')
        Prob_IS = get_Prob_IS(Dwi_ss_MNI_norm_img, ADC_ss_MNI_img,  mask_raw_MNI_img, TemplateDir)
    else:
        Prob_IS = None
        
    # get standard normalization within brainmask
    
    tmp = Dwi_ss_MNI_norm_img[mask_raw_MNI_img>0.5]
    Dwi_ss_MNI_BSN_img = (Dwi_ss_MNI_norm_img - np.mean(tmp)) / np.std(tmp)

    tmp = ADC_ss_MNI_img[mask_raw_MNI_img>0.5]
    ADC_ss_MNI_BSN_img = (ADC_ss_MNI_img - np.mean(tmp)) / np.std(tmp)
    
    # load detection model
    print('------ Loading detection model ------')
    Lesion_model = load_model(Lesion_model_name, compile=False)
    start_time = time.time()
    
    # get lesion prediction
    print('------ Inferencing lesion prediction ------')
    stroke_pred_img = get_stroke_seg_MNI(Lesion_model, 
                                         Dwi_ss_MNI_BSN_img, 
                                         ADC_ss_MNI_BSN_img, 
                                         Prob_IS, 
                                         N_channel=N_channel)
    stroke_pred_img = stroke_pred_img*mask_raw_MNI_img
    # map lesion back to raw space
    stroke_pred_raw_img = affine_map.transform_inverse((stroke_pred_img>0.5)*1, interpolation='nearest')
    stroke_pred_raw_img = stroke_pred_raw_img*mask_raw_img
    stroke_pred_raw_img = (stroke_pred_raw_img>0.5)*1
    
    stroke_pred_raw_img = remove_small_objects_InSlice(stroke_pred_raw_img)
    stroke_pred_raw_img = Stroke_closing(stroke_pred_raw_img)
    stroke_pred_raw_img = morphology.binary_fill_holes(stroke_pred_raw_img)
    
    print('------ Inferencing lesion prediction ------')
    print('It takes %.2f seconds'% (time.time() - start_time))
    start_time = time.time()
 
    # Save lesion predict
    LP_ImgJ = get_new_NibImgJ(stroke_pred_raw_img, Dwi_imgJ, dataType=np.int16)
    nib.save(LP_ImgJ, os.path.join(SubjDir, SubjID + '_' + lesion_name + '.nii.gz'))

    # save images in MNI
    if save_MNI:
        print('------ Saving images in MNI. ------')
        Dwi_ss_MNI_imgJ = get_new_NibImgJ(Dwi_ss_MNI_img, JHU_B0_ss_imgJ, dataType=np.float32)
        Dwi_ss_MNI_imgJ = MNIdePadding_imgJ(Dwi_ss_MNI_imgJ)
        nib.save(Dwi_ss_MNI_imgJ, os.path.join(SubjDir, SubjID + '_DWI_MNI.nii.gz'))
        
        B0_ss_MNI_imgJ = get_new_NibImgJ(B0_ss_MNI_img, JHU_B0_ss_imgJ, dataType=np.float32)
        B0_ss_MNI_imgJ = MNIdePadding_imgJ(B0_ss_MNI_imgJ)
        nib.save(B0_ss_MNI_imgJ, os.path.join(SubjDir, SubjID + '_b0_MNI.nii.gz'))
        
        ADC_ss_MNI_imgJ = get_new_NibImgJ(ADC_ss_MNI_img, JHU_B0_ss_imgJ, dataType=np.float32)
        ADC_ss_MNI_imgJ = MNIdePadding_imgJ(ADC_ss_MNI_imgJ)
        nib.save(ADC_ss_MNI_imgJ, os.path.join(SubjDir, SubjID + '_ADC_MNI.nii.gz'))
        
        Dwi_ss_MNI_norm_imgJ = get_new_NibImgJ(Dwi_ss_MNI_norm_img, JHU_B0_ss_imgJ, dataType=np.float32)
        Dwi_ss_MNI_norm_imgJ = MNIdePadding_imgJ(Dwi_ss_MNI_norm_imgJ)
        nib.save(Dwi_ss_MNI_norm_imgJ, os.path.join(SubjDir, SubjID + '_DWI_Norm_MNI.nii.gz'))
        
        LP_MNI_imgJ = get_new_NibImgJ(stroke_pred_img, JHU_B0_ss_imgJ, dataType=np.int16)
        LP_MNI_imgJ = MNIdePadding_imgJ(LP_MNI_imgJ)
        nib.save(LP_MNI_imgJ, os.path.join(SubjDir, SubjID + '_' + lesion_name + '_MNI.nii.gz'))
    
    if generate_report:
        print('------ Generating lesion report ------')
        ICV_vol = VolSpacing * np.sum((mask_raw_MNI_img>0.5)*1.0)
        Lesion_vol = VolSpacing * np.sum((stroke_pred_raw_img>0.5)*1.0)
        gen_lesion_report(SubjDir, SubjID, stroke_pred_img, ICV_vol, Lesion_vol, TemplateDir)
        
    if generate_result_png:
        print('------ Generating result png file ------')
        gen_result_png(SubjDir, lesion_name=lesion_name)
        
    print('------ Finish generating all results ------')
    print('It takes %.2f seconds'% (time.time() - start_time))
    
def ADS(SubjDir,
         model_name='DAGMNet_CH3',
         level_iters = [3],
         sigmas = [3.0],
         factors = [2],
         lesion_name='Lesion_Predict',
         bvalue = 1000,
         save_MNI=True,
         generate_report=True,
         generate_result_png=True
        ):
    
    CodesDir, ProjectDir, TemplateDir, TrainedNetsDir = get_DirPaths()
    SubjID = os.path.join(SubjDir,'').split('/')[-2]
    MaskNet_name =  os.path.join(TrainedNetsDir, 'BrainMaskNet.h5')
    Lesion_model_name = os.path.join(TrainedNetsDir, model_name+'.h5')
    lesion_name = model_name + '_' + lesion_name
    
    if 'CH2' in model_name:
        N_channel = 2
    elif 'CH3' in model_name:
        N_channel = 3
        
    ADS_pipeline(SubjDir,
             SubjID,
             TemplateDir=TemplateDir,
             MaskNet_name = MaskNet_name,
             Lesion_model_name = Lesion_model_name,
             N_channel = N_channel,
             level_iters = level_iters, 
             sigmas = sigmas, 
             factors = factors, 
             lesion_name=lesion_name,
             bvalue = bvalue,
             save_MNI=save_MNI,
             generate_report=generate_report,
             generate_result_png=generate_result_png
            )
    print('------ Process Done !!! ------')