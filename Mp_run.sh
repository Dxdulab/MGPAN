#!/bin/bash

#JSUB -J NielsenHB_2014-F1MP8+BCE+1cl+tymeta


#IjazUZ_2017
#HallAB_2017
#NielsenHB_2014
#HMP_2019_ibdmdb
#V2018+L2016
#VilaAV_2018
#WirbelJ_2018
#VogtmannE_2016
#MetaCardis_2020_a

#JSUB -q gpu


#JSUB -gpgpu "1 type=NVIDIAGeForceRTX3080"
# JSUB -m gpu07
#NVIDIAA100-PCIE-40GB
#TeslaV100-SXM2-32GBmod
#NVIDIAGeForceRTX3080

#JSUB -cwd /home/24031212340/MGPAN

#JSUB -e Mp_Error/error.%J
#JSUB -o Mp_Output/output.%J

source /apps/software/anaconda3/etc/profile.d/conda.sh

conda activate MGP2 
#module load cuda/11.3

nvidia-smi
# nvcc --version

#CUDA_LAUNCH_BLOCKING=1 
#export CUDA_LAUNCH_BLOCKING=1 
#python main_metapath_Gp_typeaware.py
python main.py
# python main_metapath_GP_microbe_only.py
# python main_metapath_edge_ablation.py
#python main_metapath_pairwise_bimodal_ablation.py
#python main_metapath_GP_wo_mp_attention2.py
#main_cross1.py
#main_metapath_KF.py
#python test.py

#jsub < Mp_run.sh
#jctrl kill -f