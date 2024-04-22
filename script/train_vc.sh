#!/bin/bash -l

basepath=../bnn
WL=8
eta=0.15
epoch=235
weight_decay=1e-3
dynamic=hmc2
data=CIFAR100
quant=vc
acc=-1
u=5
r=5
seed=1
type=decay
temperature=0.001
weight_type=block
grad_type=block
vel_type=block
scale_x=false
num_savemodel=85

<<EOT
#!/bin/bash -l


python $basepath/train_bnn.py --start_epoch 0 --save_epoch $epoch --epochs $epoch --lr_init $eta --wd $weight_decay \
       --inverse_mass $u --friction $r --lr_type $type --dataset $data --quant_type $quant --quant_acc $acc --wandb_log false \
       --weight-type $weight_type --grad-type $grad_type --vel-type $vel_type --activate-type $weight_type --error-type $weight_type \
       --wl-weight ${WL} --wl-grad ${WL} --fl-weight ${WL} --wl-error ${WL} --wl-activate ${WL} \
       --fl-grad ${WL} --dynamic $dynamic --pretrain_path none --intermediate true --seed $seed --temperature $temperature --scale_x ${scale_x} \
       --wl-vel ${WL} --fl-vel ${WL}

EOT

