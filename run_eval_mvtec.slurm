#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg4|dlg5

#SBATCH --job-name="e_mvtec"
#SBATCH --account=punim1623
#SBATCH --time=0-02:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --mem=60G

#### export WORLD_SIZE=2   ### FIXME: update world size: nodes x ntasks-per-node
#### export MASTER_PORT=28400
#### echo ">>> NODELIST="${SLURM_NODELIST}
#### master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#### export MASTER_ADDR=$master_addr
#### echo ">>> MASTER_ADDR="$MASTER_ADDR

module purge

eval "$(conda shell.bash hook)"
conda activate anogpt


python examples/musc_main.py \
  --device 0 \
  --data_path ./data/mvtec_anomaly_detection/ \
  --dataset_name mvtec_ad \
  --class_name ALL \
  --backbone_name ViT-L-14-336 \
  --pretrained openai \
  --feature_layers 5 11 17 23 \
  --img_resize 518 \
  --divide_num 1 \
  --r_list 1 3 5 \
  --batch_size 4 \
  --output_dir ./output \
  --vis True \
  --save_excel True \
  --online \

##Log this job's resource usage stats###
my-job-stats -a -n -s
##