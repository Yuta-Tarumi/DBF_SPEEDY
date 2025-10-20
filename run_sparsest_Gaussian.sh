#!/bin/bash                                                                                                                                                                      
#SBATCH -p all
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH -J ytarumi-sparsest-gaussian
#SBATCH -o stdout/stdout.%J
#SBATCH -e stderr/stderr.%J

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# DBF training
source ${HOME}/py312_dbf/bin/activate
rm -r /home/ytarumi/DBF_work/memlogs
mkdir /home/ytarumi/DBF_work/memlogs
python train_script.py  --config config/decoder_transformer_dim8192_sparsest_Gaussian.yaml --niter 0
python train_script.py  --config config/decoder_transformer_dim2048_sparsest_Gaussian.yaml --niter 0
python train_script.py  --config config/decoder_transformer_dim512_sparsest_Gaussian.yaml --niter 0
python train_script.py  --config config/decoder_transformer_dim128_sparsest_Gaussian.yaml --niter 0
deactivate

RETCODE=$?
exit ${RETCODE}
