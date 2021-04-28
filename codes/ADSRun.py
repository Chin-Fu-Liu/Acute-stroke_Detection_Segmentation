# Copyright (c) 2021, Andreia V. Faria, Chin-Fu Liu
# All rights reserved.
#
# This program is free software; 
# you can redistribute it and/or modify it under the terms of the GNU General Public License v3.0. 
# See the LICENSE file or read the terms at https://www.gnu.org/licenses/gpl-3.0.en.html.


import sys
import os
import argparse

from ADS_bin import *

OPT_MODEL = "-model"
OPT_INPUTFOLDER = "-input"
OPT_BVALUE = "-bvalue"
OPT_SAVEMNI = "-save_MNI"
OPT_GENMASK = "-generate_brainmask"
OPT_GENREPORT = "-generate_report"
OPT_GENRESULTPNG = "-generate_result_png"

def get_arg_parser():
    parser = argparse.ArgumentParser(prog='ADSRun', formatter_class=argparse.RawTextHelpFormatter,
    description="\n This software allows creation of lesion predicts for acute ischemic subjects in biomedical NIFTI GZ volumes.\n"+\
                "The project is hosted at: https://github.com/Chin-Fu-Liu/Acute_Stroke_Detection/ \n"+\
                "See the documentation for details on its use.\n"+\
                "For questions and feedback, please contact:cliu104@jhu.edu")
    
    parser.add_argument(OPT_INPUTFOLDER, dest='input',  type=str, 
                        help='Specify the subject input folder. FolderName should be SubjID and  the naming and format of images under subject folder should fulfill the requirements in biomedical NIFTI GZ volumes as  https://github.com/Chin-Fu-Liu/Acute_Stroke_Detection/')
    
    parser.add_argument(OPT_MODEL, dest='model', type=str, default='DAGMNet_CH3',
                        help='Specify which the trained model to be used, like DAGMNet_CH3, DAGMNet_CH2, UNet_CH3, UNet_CH2, FCN_CH3, FCN_CH2. Models should be under the Trained_Networks folder with the same naming. (default: DAGMNet_CH3)')

    parser.add_argument(OPT_BVALUE, dest='bvalue', type=int, default=1000,
                        help='Specify the b-value to calculate ADC, if ADC is not given. If ADC is given under subjectID folder, this option will be ignored. (default: 1000)')
    
    parser.add_argument(OPT_SAVEMNI, dest='save_MNI', type=bool, default=True,
                        help='For saving all images in MNI, please set this option as True, otherwise set it as False. (default: True)')

    parser.add_argument(OPT_GENMASK, dest='generate_brainmask', type=bool, default=False,
                        help='For generating brain mask, please set this option as True, otherwise set it as False. (default: False)')
    
    parser.add_argument(OPT_GENREPORT, dest='generate_report', type=bool, default=True,
                        help='For generating lesion report in txt, please set this option as True, otherwise set it as False. (default: True)')
    
    parser.add_argument(OPT_GENRESULTPNG, dest='generate_result_png', type=bool, default=True,
                        help='For generating result in png, please set this option as True, otherwise set it as False. (default: True)')
    
    args = parser.parse_args()
    return args

def main(args):
    SubjDir = args.input
    model_name = args.model
    bvalue = args.bvalue
    save_MNI = args.save_MNI
    generate_brainmask = args.generate_brainmask
    generate_report = args.generate_report
    generate_result_png = args.generate_result_png,
    
    ADS(SubjDir=SubjDir,
         model_name=model_name,
#          level_iters = [3],
#          sigmas = [3.0],
#          factors = [2],
#          lesion_name='Lesion_Predict'
         bvalue = bvalue,
         save_MNI=save_MNI,
         generate_brainmask=generate_brainmask,
         generate_report=generate_report,
         generate_result_png=generate_result_png
        )

if __name__ == "__main__":       
    args = get_arg_parser()
#     print(args)
    if len(sys.argv) == 1:
        print("For help on the usage of ADSRun, please use the option -h."); exit(1)
    if not args.input:
        print("ERROR: Option ["+OPT_INPUTFOLDER+"] must be specified.\n"+\
              "Please try [-h] for more information. Exiting."); exit(1)
    main(args)