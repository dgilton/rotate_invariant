#!/bin/bash

#SBATCH --account=pi-foster  # default pi-chard
#SBATCH --job-name=clouds
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --partition=gm4
##SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6000
#SBATCH --time=1:00:00
##SBATCH --qos=gm4

module load cuda/8.0
module load Anaconda3/5.0.0.1
source activate py3501

cwd=`pwd`

# directory
logdir=${cwd}/log
figdir=${cwd}/fig
output_modeldir=${cwd}/output_model/${SLURM_JOB_ID}
input_datadir="/project2/chard/clouds/data/MOD02/MOD02/clouds_laads_preprocessed_2000_2018_band28_29_31_normed"

# params
learning_rate=0.01 # Adam:0.001 SGD:0.01
num_epoch=5  # 96min for 25 epochs on one-V100
batch_size=32
copy_size=4
dangle=1
c_lambda=0.5
height=128
width=128
nblocks=5
save_every=1

python train_clouds_fully_connected.py \
  --logdir ${logdir} \
  --figdir ${figdir} \
  --input_datadir ${input_datadir} \
  --output_modeldir ${output_modeldir} \
  --lr ${learning_rate} \
  --num_epoch ${num_epoch} \
  --batch_size ${batch_size} \
  --copy_size=${copy_size} \
  --dangle=${dangle} \
  --c_lambda ${c_lambda} \
  --height ${height} \
  --width ${width} \
  --nblocks ${nblocks} \
  --save_every ${save_every} \
  --expname ${SLURM_JOB_ID} \
  --debug


cp -r ${cwd}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}* ${output_modeldir}/
