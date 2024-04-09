import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
# import tabulate
#from qtorch.quant import quantizer
#import qtorch
from qtorch.quant import *
from torch.optim import SGD
# from torch.optim.lr_scheduler import LambdaLR
from qtorch import BlockFloatingPoint, FixedPoint, FloatingPoint
from qtorch.auto_low import sequential_lower
import logging
import os
from utils import Saver
import glob
import math

num_types = ["weight", "activate", "grad", "error", "momentum"]

parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument(
    "--dataset", type=str, default="MNIST", help="dataset name: MNIST"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="../data",
    metavar="PATH",
    help='path to datasets location (default: "./data")',
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size (default: 128)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    metavar="N",
    help="number of epochs to train (default: 300)",
)
parser.add_argument(
    "--lr_init",
    type=float,
    default=0.05,
    metavar="LR",
    help="initial learning rate (default: 0.01)",
)
parser.add_argument(
    "--wd", type=float, default=1e-4, help="weight decay (default: 1e-4)"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="N", help="random seed (default: 1)"
)

parser.add_argument(
    "--wl_weight",
    type=int,
    default=8,
    metavar="N",
    help="word length in bits for {}; -1 if full precision.",
)
parser.add_argument(
    "--rounding",
    type=str,
    default="stochastic",
    metavar="S",
    choices=["stochastic", "nearest"],
    help="rounding method for {}, stochastic or nearest",
)
parser.add_argument(
    "--model", type=str, help="Logistic or MLP"
)
parser.add_argument(
    "--noise", type=bool, default=True, metavar="N", help="whether to add SGLD noise"
)
parser.add_argument(
    "--momentum", type=float, default=0., metavar="N", help="momentum"
)
parser.add_argument(
    "--temperature", type=float, default=1., metavar="N", help="temperature"
)
parser.add_argument(
    "--run", type=int, default=-1, metavar="N", help="run number"
)
parser.add_argument(
    "--quant_acc", type=int, default=-1, metavar="N", help="accumulator quant num"
)
parser.add_argument(
    "--quant_type", type=str, default="vc", help="quant type"
)
parser.add_argument(
    "--dynamic", type=str, default='sgld', help='dynamic sgld or hmc'
)
parser.add_argument(
    "--weight_acc", type=str, default='low', help="save a full or low precision parameter"
)
parser.add_argument(
    "--grad_acc", type=str, default='low', help='precision of gradient'
)
parser.add_argument(
    "--inverse_mass", type=float, default=1, help="inverse mass"
)
parser.add_argument(
    "--friction", type=float, default=2, help="friction"
)
parser.add_argument(
    "--report_freq", type=int, default=25, help='Report Frequency'
)
parser.add_argument(
    "--checkname", type=str, default="train", help='Mode'
)
parser.add_argument(
    "--scale_x", type=str, default='false'
)
parser.add_argument(
    "--num_savemodel", type=int, default=50
)
parser.add_argument(
    "--FL", type=int, default=6
)

# args = parser.parse_args(
#     ['--quant_type', 'vc', '--quant_acc', '-1', '--scale_x', 'true', '--dynamic', 'hmc2', '--model', 'MLP',
#      '--lr_init', '0.01', '--FL', '8'])

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
utils.set_seed(args.seed, args.cuda)
print("using costomized optim")
from optim_nom import OptimLP
loaders = utils.get_data(args.dataset, args.data_path, args.batch_size, num_workers=0)
num_classes = utils.num_classes_dict[args.dataset]
criterion = F.cross_entropy
print('quant_type',args.quant_type,'noise:',args.noise,'t',args.temperature,'wl',args.wl_weight,'quant_acc:',args.quant_acc,'seed:',args.seed)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classifier = nn.Linear(28*28, 10)
        # self.classifier.weight.data.fill_(0.)
        # self.classifier.bias.data.fill_(0.)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.classifier(x)
        return x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(28*28, 100)
        self.layer2 = nn.Linear(100, 10)

    def forward(self, x):
        out = x.view(-1, 28*28)
        out = F.relu(self.layer1(out))
        out = self.layer2(out)
        return out


def compute_rate(u, r, eta):
    return u/r**2*(r*eta+math.exp(-r*eta)-1)


if args.wl_weight >0:
    number_dict = dict()
    FL = args.FL

    num_wl = args.wl_weight
    number_dict = FixedPoint(wl=num_wl, fl=FL)
    print("{}".format(number_dict))
    quant_dict = quantizer(
        forward_number=number_dict, forward_rounding=args.rounding
    )
    if args.model == 'logistic':
        model = Net()
    elif args.model == 'MLP':
        model = MLP()
    model = sequential_lower(
        model,
        layer_types=["activation"],
        forward_number=number_dict,
        backward_number=number_dict,
        forward_rounding=args.rounding,
        backward_rounding=args.rounding,
    )
    # model.classifier = model.classifier[0]  # removing the final quantization module
    model.cuda()
    if args.quant_acc > 0:
        acc_num = BlockFloatingPoint(wl=args.quant_acc, dim=0)
        quant_acc = quantizer(
            forward_number=acc_num, forward_rounding=args.rounding
        )
    elif args.quant_acc == -2:
        quant_acc = "full"
    else:
        quant_acc = None
    if args.grad_acc == 'low':
        grad_quant = quant_dict
    else:
        grad_quant = None
    if args.weight_acc == "full" and args.grad_acc=='full':
        weight_quant = None
    else:
        weight_quant = quant_dict

    optimizer = SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.wd)
    optimizer = OptimLP(
        optimizer,
        weight_quant=quant_dict,
        grad_quant=grad_quant,
        momentum_quant=weight_quant,
        acc_quant=quant_acc,
        noise=args.noise,
        temperature=args.temperature,
        datasize=60000,
        WL=args.wl_weight,
        FL=FL,
        quant_type=args.quant_type,
        # gamma, mu, eta
        inverse_mass=args.inverse_mass,
        friction=args.friction,
        dynamic=args.dynamic,
        scale_x=args.scale_x
        # number_type='block'
    )
else:
    model = Net()
    model.cuda()
    optimizer = SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.wd)
    optimizer = OptimLP(
        optimizer,
        noise=args.noise,
        temperature=args.temperature,
        datasize=60000,
        # gamma, mu, eta
        inverse_mass=args.inverse_mass,
        friction=args.friction,
        eta_rate=args.eta_rate
    )
# Prepare logging
columns = [
    "ep",
    "lr",
    "tr_loss",
    "tr_acc",
    "tr_time",
    "te_loss",
    "te_acc",
]
# model.load_state_dict(torch.load('checkpoints/%s_%s_%d_cycle_oldoptim_e%d_noise%s_acc%d_wd%s_lr%s_m%s_M%d_t%s_seed%d_%i.pt'%(args.dataset,args.model,args.wl_weight,args.epochs,args.noise,args.quant_acc,args.wd,0.08,args.momentum,args.M,args.temperature,args.seed,4)))
saver = Saver(args)
saver.create_exp_dir(scripts_to_save=glob.glob('*.py') + glob.glob('*.sh') + glob.glob('*.yml'))
saver.save_experiment_config()

summary = utils.TensorboardSummary(saver.experiment_dir)
writer = summary.create_summary()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p',
                    filename=os.path.join(saver.experiment_dir, 'log.txt'), filemode='w')
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# logging.getLogger().addHandler(console)

# if 'sgld' in args.quant_type:
#     args.eta_rate = compute_rate(args.inverse_mass, args.friction, args.eta_rate)

mt = 0
for epoch in range(args.epochs):
    # lr = utils.schedule(epoch, args.lr_init, args.swa_start, args.swa_lr)
    # utils.adjust_learning_rate(optimizer, lr)
    logging.info('--------- Epoch: {} ----------'.format(epoch))
    train_res = utils.run_epoch(args,
        loaders["train"], model, criterion, epoch, logfile=logging, writer=writer, optimizer=optimizer, phase="train"
    )
    test_res = utils.run_epoch(args, loaders["test"], model, criterion, epoch, logfile=logging, writer=writer, phase="eval")

    values = [
        epoch + 1,
        optimizer.param_groups[0]["lr"],
        *train_res.values(),
        *test_res.values(),
    ]
    utils.print_table(columns, values, epoch, logfile=logging)


    if args.noise and epoch>args.epochs-args.num_savemodel:
        print('save!')
        torch.save(model.state_dict(), os.path.join(saver.experiment_checkpoints, 'model_{}.pt'.format(mt)))
                   # 'checkpoints/reg_%s_%d_noise%s_acc%d_type%s_t%s_seed%d_%i.pt'%(args.dataset,args.wl_weight,args.noise,args.quant_acc,args.quant_type,args.temperature,args.seed,mt))
        mt += 1
if not args.noise:
    print('save!')
    torch.save(model.state_dict(), os.path.join(saver.experiment_checkpoints, 'checkpoints', 'model_{}.pt'.format(mt)))
    # torch.save(model.state_dict(),'checkpoints/reg_%s_%d_noise%s_acc%d_seed%d_%i.pt'%(args.dataset,args.wl_weight,args.noise,args.quant_acc,args.seed,mt))

