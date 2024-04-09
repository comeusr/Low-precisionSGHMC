import os
import torch
import tabulate
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from time import perf_counter
import glob
import shutil
from collections import OrderedDict
import numpy as np
from tensorboardX import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed, cuda):
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def run_epoch(args, loader, model, criterion, epoch, logfile, writer, optimizer=None, phase="train"):
    assert phase in ["train", "eval"], "invalid running phase"
    loss_sum = 0.0
    correct = 0.0

    if phase == "train":
        model.train()
    elif phase == "eval":
        model.eval()

    ttl = 0
    start = perf_counter()
    with torch.autograd.set_grad_enabled(phase == "train"):
        for i, (input, target) in enumerate(loader):
            input = input.to(device=device)
            target = target.to(device=device)
            output = model(input)
            loss = criterion(output, target)
            loss_sum += loss.cpu().item() * input.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            ttl += input.size()[0]

            if phase == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(epoch)

            if (i + 1) % args.report_freq == 0 and phase == 'train':
                with torch.no_grad():
                    check_correct = pred.eq(target.data.view_as(pred)).sum()
                    check_total = input.size()[0]
                    check_acc = check_correct / check_total
                logfile.info(
                    'Epoch: [{Epoch}][{iter}\{len}]\t Loss: {loss:.2f} Train Acc: {acc:.2f}'.format(Epoch=epoch,
                                                                                                    iter=i + 1,
                                                                                                    len=len(loader),
                                                                                                    loss=loss.data,
                                                                                                    acc=check_acc))
                writer.add_scalar('{}_Loss/Iter'.format(phase), loss.data, i + epoch * len(loader))
                writer.add_scalar('{}_Acc/Iter'.format(phase), check_acc, i + epoch * len(loader))


    elapse = perf_counter() - start
    correct = correct.cpu().item()
    res = {
        "loss": loss_sum / float(ttl),
        "accuracy": correct / float(ttl) * 100.0,
    }
    # if phase == "eval":
    #     writer
    if phase == "train":
        res["train time"] = elapse

    return res


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 1.0 - alpha
        param1.data += param2.data * alpha


def print_table(columns, values, epoch, logfile):
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 40 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)
    logfile.info(table)


num_classes_dict = {
    "CIFAR10": 10,
    "CIFAR100": 100,
    "MNIST": 10
}


def get_data(dataset, data_path, batch_size, num_workers, train_shuffle=True):
    print("Loading dataset {} from {}".format(dataset, data_path))
    ds = getattr(datasets, dataset.upper())
    path = os.path.join(data_path)
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    )
    train_set = ds(path, train=True, download=True, transform=transform_train)
    test_set = ds(path, train=False, download=True, transform=transform_test)

    loaders = {
        "train": torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=train_shuffle,
            num_workers=0
        ),
        "test": torch.utils.data.DataLoader(
            test_set,
            batch_size=1000,
            shuffle=False,
            num_workers=0
        ),
    }

    return loaders


class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('/logistics', args.dataset, args.model, args.checkname)
        self.runs = glob.glob(os.path.join(self.directory, 'experiment_*'))
        # run_list = sorted([int(m.split('_')[-1]) for m in self.runs])
        # run_id = run_list[-1] + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory,
                                           '{dynamic}_{type}_{acc}_eta_{eta}_reg_{bits}_noise_{noise}_t{t}_seed{seed}_u{u}_gamma{gamma}_wd{wd}'.format(
                                               dynamic=args.dynamic, eta=args.lr_init, bits=args.FL, noise=args.noise,
                                               acc=args.quant_acc, type=args.quant_type, t=args.temperature,
                                               seed=args.seed, u=args.inverse_mass, gamma=args.friction, wd=args.wd))
        self.experiment_checkpoints = os.path.join(self.experiment_dir, 'checkpoints')
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        if not os.path.exists(self.experiment_checkpoints):
            os.makedirs(self.experiment_checkpoints)


    def create_exp_dir(self, scripts_to_save=None):
        print('Experiment dir : {}'.format(self.experiment_dir))
        if scripts_to_save is not None:
            os.mkdir(os.path.join(self.experiment_dir, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(self.experiment_dir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        with open(logfile, 'w+') as log_file:
            p = OrderedDict()
            p['dataset'] = self.args.dataset
            p['seed'] = self.args.seed
            p['epochs'] = self.args.epochs
            p['eta'] = self.args.lr_init
            p['bits'] = self.args.wl_weight
            p['noise'] = self.args.noise
            p['temperature'] = self.args.temperature
            p['inverse mass'] = self.args.inverse_mass
            p['friction'] = self.args.friction
            p['quant type'] = self.args.quant_type
            p['quant acc'] = self.args.quant_acc

            for key, val in p.items():
                log_file.write(key + ':' + str(val) + '\n')


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer
