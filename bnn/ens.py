import sys
sys.path.append("..")
import argparse
import torch
import torch.nn.functional as F
import utils
import models
from qtorch.quant import *
from qtorch import BlockFloatingPoint, FixedPoint, FloatingPoint
import numpy as np
import os
from utils import Saver
num_types = ["weight", "activate", "grad", "error"]

parser = argparse.ArgumentParser(description="testing")
parser.add_argument(
    "--dataset", type=str, default="CIFAR100", help="dataset name: CIFAR10 or CIFAR100"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="../data",
    metavar="PATH",
    help='path to datasets location (default: "../data")',
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size (default: 128)",
)
parser.add_argument(
    "--model",
    type=str,
    default="ResNet18LP",
    # required=True,
    metavar="MODEL",
    help="model name (default: None)",
)
print('test')
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    help="number of epochs to train (default: 200)",
)
parser.add_argument(
    "--eval_freq",
    type=int,
    default=5,
    help="evaluation frequency (default: 5)",
)
parser.add_argument(
    "--lr_init",
    type=float,
    default=0.5,
    help="initial learning rate (default: 0.01)",
)
parser.add_argument(
    "--wd", type=float, default=5e-4, help="weight decay (default: 1e-4)"
)
parser.add_argument(
    "--seed", type=int, default=1, help="random seed (default: 1)"
)
for num in num_types:
    parser.add_argument(
        "--wl-{}".format(num),
        type=int,
        default=-1,
        help="word length in bits for {}; -1 if full precision.".format(num),
    )
    parser.add_argument(
        "--fl-{}".format(num),
        type=int,
        default=-1,
        help="number of fractional bits for {}; -1 if full precision.".format(num),
    )
    parser.add_argument(
        "--{}-man".format(num),
        type=int,
        default=-1,
        help="number of bits to use for mantissa of {}; -1 if full precision.".format(num),
    )
    parser.add_argument(
        "--{}-exp".format(num),
        type=int,
        default=-1,
        help="number of bits to use for exponent of {}; -1 if full precision.".format(num),
    )
    parser.add_argument(
        "--{}-type".format(num),
        type=str,
        default="full",
        choices=["fixed", "block", "float", "full", "hfloat"],
        help="quantization type for {}; fixed or block.".format(num),
    )
    parser.add_argument(
        "--{}-rounding".format(num),
        type=str,
        default="stochastic",
        choices=["stochastic", "nearest"],
        help="rounding method for {}, stochastic or nearest".format(num),
    )
parser.add_argument(
    "--noise", type=bool, default=True,
    help="whether to add Gaussian noise in the update. True: SGLD or SGHMC, False: SGD"
)
parser.add_argument(
    "--temperature", type=float, default=0.001, help="temperature"
)
parser.add_argument(
    "--quant_acc", type=int, default=-1, help="accumulator precision. -1: low-precision, -2: full-precision"
)
parser.add_argument(
    "--quant_type", type=str, default="vc", help="quant type"
)
parser.add_argument(
    "--num_savemodel", type=int, default=30, help="number of saved models"
)
parser.add_argument(
    "--lr_type", type=str, default='decay', help="LR schedule"
)
parser.add_argument(
    "--M", type=int, default=7, help="number of cycles in cyclical LR schedule"
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
    '--dynamic', type=str, default='hmc2', help='Dynamic'
)
parser.add_argument(
    '--pretrain_path', type=str, default='none', help='Pretrained Network Path'
)
parser.add_argument(
    '--save_epoch', type=int, default=240, help='If save, save the model at this epoch'
)
parser.add_argument(
    '--intermediate', type=str, default='false', help='Output intermediate model or not'
)
parser.add_argument(
    '--start_epoch', type=int, default=0, help='Continue to train the model from this epoch'
)
parser.add_argument(
    '--wandb_log', type=str, default='false', help="Wandb log to debug or not."
)
parser.add_argument(
    '--momentum_path', type=str
)
parser.add_argument(
    '--scale_path', type=str
)
parser.add_argument(
    "--scale_x", type=str, default='false'
)
parser.add_argument(
    '--test_freq', type=int, default=2
)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
utils.set_seed(args.seed, args.cuda)

loaders = utils.get_data(args.dataset, args.data_path, args.batch_size, num_workers=0)
saver = Saver(args)

def make_number(number, wl=-1, fl=-1, exp=-1, man=-1):
    if number == "fixed":
        return FixedPoint(wl, fl)
    elif number == "block":
        return BlockFloatingPoint(wl,dim=0)
    elif number == "float":
        return FloatingPoint(exp, man)
    elif number == "full":
        return FloatingPoint(8, 23)
    else:
        raise ValueError("invalid number type")

if args.wl_weight >0:
    number_dict = {}
    for num in num_types:
        num_wl = getattr(args, "wl_{}".format(num))
        num_fl = getattr(args, "fl_{}".format(num))
        num_type = getattr(args, "{}_type".format(num))
        num_rounding = getattr(args, "{}_rounding".format(num))
        num_man = getattr(args, "{}_man".format(num))
        num_exp = getattr(args, "{}_exp".format(num))
        number_dict[num] = make_number(
            num_type, wl=num_wl, fl=num_fl, exp=num_exp, man=num_man
        )
        print("{:10}: {}".format(num, number_dict[num]))

    weight_quantizer = quantizer(
        forward_number=number_dict["weight"], forward_rounding=args.weight_rounding
    )
    grad_quantizer = quantizer(
        forward_number=number_dict["grad"], forward_rounding=args.grad_rounding
    )
    acc_err_quant = lambda: Quantizer(
        number_dict["activate"],
        number_dict["error"],
        args.activate_rounding,
        args.error_rounding,
    )

    # Build model
    print("Model: {}".format(args.model))
    model_cfg = getattr(models, args.model)
    model_cfg.kwargs.update({"quant": acc_err_quant})

    if args.dataset == "CIFAR10":
        num_classes = 10
        ds = 50000
        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
        criterion = F.cross_entropy
    elif args.dataset == "CIFAR100":
        num_classes = 100
        ds = 50000
        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
        criterion = F.cross_entropy
    elif args.dataset == "IMAGENET12":
        num_classes = 1000
    model.cuda()
    if args.quant_acc == -2:
        quant_acc = "full"
    else:
        quant_acc = None
else:
    print("Model: {}".format(args.model))
    model_cfg = getattr(models, args.model)
    if args.dataset == "CIFAR10":
        num_classes = 10
        ds = 50000
        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
        criterion = F.cross_entropy
    elif args.dataset == "CIFAR100":
        num_classes = 100
        ds = 50000
        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
        criterion = F.cross_entropy
    elif args.dataset == "IMAGENET12":
        num_classes = 1000
    model.cuda()

def run_epoch(loader, model, criterion, optimizer=None, phase="train"):
    assert phase in ["train", "eval"], "invalid running phase"
    loss_sum = 0.0
    correct = 0.0

    if phase=="train": model.train()
    elif phase=="eval": model.eval()

    ttl = 0
    pred_list = []
    truth_res = []
    with torch.autograd.set_grad_enabled(phase=="train"):
        for i, (input, target) in enumerate(loader):
            target = target.cuda()
            truth_res += list(target.data)
            input = input.cuda()
            output = model(input)
            pred = output.data.max(1, keepdim=True)[1]  
            pred_list.append(F.softmax(output,dim=1))
            loss = criterion(output, target)
            loss_sum += loss.cpu().item() * target.size(0)
            correct += pred.eq(target.data.view_as(pred)).sum()
            ttl += target.size(0)

    correct = correct.cpu().item()
    acc = correct / float(ttl)
    print('loss', loss_sum / float(ttl),'accuracy', acc,)
    pred_list = torch.cat(pred_list,0)
    return pred_list,truth_res,acc
   

def get_accuracy(truth, pred):
    assert len(truth)==len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i]==pred[i]:
             right += 1.0
    return right/len(truth)

def expected_calibration_error(y_true, y_pred, num_bins=10):
    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true).astype(np.float32)
    prob_y = np.max(y_pred, axis=-1)

    b0 = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins=b0, right=True)

    o = 0
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            temp = np.sum(correct[mask] - prob_y[mask])
            o += np.abs(temp)

    return o / y_pred.shape[0]

pred_list = []
args.test_freq = 1

count = 0
for mt in range(args.num_savemodel):
    if mt % args.test_freq == 0:
        count += 1
        model_path = os.path.join(saver.experiment_dir, 'checkpoints', 'model_{}.pt'.format(mt))
        model.load_state_dict(torch.load(model_path))
        pred, truth_res, acc = run_epoch(loaders['test'], model, criterion,
                                    optimizer=None, phase="eval")

        truth_res = torch.as_tensor(truth_res)
        ece = expected_calibration_error(truth_res.cpu().numpy(), pred.cpu().numpy())
        print('ECE: {}'.format(ece*100))

        pred_list.append(pred)


fake = sum(pred_list)/(count)
values, pred_label = torch.max(fake, dim=1)
pred_res = list(pred_label.data)
acc = get_accuracy(truth_res, pred_res)
print('error%', (1-acc)*100)
truth_res = torch.as_tensor(truth_res)
ece = expected_calibration_error(truth_res.cpu().numpy(), fake.data.cpu().numpy())
print('ece%', ece*100)