#!/bin/bash -l

basepath=../bnn
WL=8
eta=0.15
epoch=245
weight_decay=1e-3
dynamic=hmc2
data=CIFAR100
quant=naive
acc=-1
u=5
r=2
seed=1
type=decay
temperature=0.001
weight_type=block
grad_type=block
scale_x=false
num_savemodel=85


sbatch --time=4:00:00 --nodes=1 --gpus-per-node=1 <<EOT
#!/bin/bash -l

#SBATCH --output /home/wang4538/lowPrecisionHMC/bnn/out/%j.out
#SBATCH --error /home/wang4538/lowPrecisionHMC/bnn/out/%j.out

python $basepath/train_bnn.py --start_epoch 0 --save_epoch 245 --epochs $epoch --lr_init $eta --wd $weight_decay \
       --inverse_mass $u --friction $r --lr_type $type --dataset $data --quant_type $quant --quant_acc $acc --wandb_log false \
       --weight-type $weight_type --grad-type $grad_type --wl-weight ${WL} --wl-grad ${WL} --fl-weight ${WL} \
       --fl-grad ${WL} --dynamic $dynamic --pretrain_path none --intermediate false --seed $seed --temperature $temperature --scale_x ${scale_x} \
       --num_savemodel $num_savemodel

EOT



