import torch
from qtorch.quant import quantizer
from qtorch import BlockFloatingPoint, FixedPoint, FloatingPoint
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, cauchy, t
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
import math
import argparse
import pandas as pd
from scipy.stats import wasserstein_distance

parser = argparse.ArgumentParser(description='Mixture Gaussian Synthetic')
parser.add_argument(
    '--nsample', type=int, default=50000, help='Number of Sample'
)
parser.add_argument(
    '--lr', type=float, default=0.1, help='learning rate'
)
parser.add_argument(
    '--nstep', type=int, default=10, help='Step before Sample'
)
parser.add_argument(
    '--seed', type=int, default=10, help='random seed'
)
parser.add_argument(
    '--WL', type=int, default=8
)
parser.add_argument(
    '--FL', type=int, default=3
)
parser.add_argument(
    '--gamma', type=float, default=2
)
parser.add_argument(
    '--precision', type=str, default='full', help='Type of low Precision Algorithm'
)
parser.add_argument(
    '--dynamic', type=str, default='hmc', help='Dynamic: HMC or SGLD'
)
parser.add_argument(
    '--type', type=str, default='mix', help='mix Gaussian or Single Gaussian'
)
parser.add_argument(
    '--algorithm', type=str, default='Q_x_eta_Q_v'
)
parser.add_argument(
    '--sigma', type=float, default=0.0001, help='Gradient Noise'
)
parser.add_argument(
    '--U', type=float, default=2, help='Inverse Mass Parameter'
)

parser.add_argument(
    '--dimension', type=int, default=1
)
parser.add_argument(
    '--batch_size', type=int, default=5000
)

args = parser.parse_args()
# args = parser.parse_args(['--precision', 'low_full', '--dynamic', 'sgld', '--U', '4', '--nsample', '1000000'])


rcParams.update({'figure.autolayout': True})
import os
if not os.path.exists('figs'):
    os.makedirs('figs')

def Q_vc_sgld(mu, var):
    if var>VAR_F:
        x = mu + (var-VAR_F)**.5*torch.randn(1)
        quant_x = quant_n(x)
        residual = x - quant_x
        theta = quant_x + torch.sign(residual)*sample_mu_sgld(torch.abs(residual))
    else:
        quant_mu = quant_s(mu)
        residual = mu - quant_mu
        p1 = torch.abs(residual)/D
        var_s = (1.-p1)*residual**2+p1*(-residual+torch.sign(residual)*D)**2
        if var>var_s:
            theta = quant_mu + sample_sgld(var-var_s)
        else:
            theta = quant_mu #this line should not be used often, otherwise the variance will be larger than the truth
    theta = torch.clamp(theta, min=-2**(WL-FL-1), max=2**(WL-FL-1)-2**(-FL))
    return theta


def sample_sgld(var):
    p1 = var/(2*D**2)
    p2 = p1
    u = torch.rand(1)
    if u<p1:
        return D 
    elif u<p1+p2:
        return -D 
    else:
        return 0

def sample_mu_sgld(mu):
    p1 = (VAR_F+mu**2+mu*D)/(2*D**2)
    p2 = (VAR_F+mu**2-mu*D)/(2*D**2)
    u = torch.rand(1)
    if u<p1:
        return D 
    elif u<p1+p2:
        return -D 
    else:
        return 0


# Generate dx and dv
def brown(u, gamma, eta, d):
    xi = np.random.normal(0, 1, size=(1, d))
    var1 = u * (1 - math.exp(-2 * gamma * eta))
    var2 = (u / gamma ** 2) * (2 * gamma * eta + 4 * math.exp(-gamma * eta) - math.exp(-2 * gamma * eta) - 3)
    corr = u / gamma * (1 - 2 * math.exp(-gamma * eta) + math.exp(-2 * gamma * eta))
    xi_v = xi * math.sqrt(var1)
    xi_x = corr / var1 * xi_v + math.sqrt(var2 - corr ** 2 / var1) * np.random.normal(0, 1, size=(1, d))

    return torch.tensor(xi_v), torch.tensor(xi_x)



def pos_grad(theta, x):
    sigma_2 = torch.tensor(2)
    items = np.random.choice(range(x.shape[0]), size=64)
    grad = torch.mean(theta - x[items]) / torch.square(sigma_2)
    return grad


# Update function
def SGULMCMC_theta_update_pos(w, u, gamma, eta, d, w_v, sample, Low=True):
    if Low == False:
        grad_value = pos_grad(w, sample)
    else:
        grad_temp = pos_grad(quant_s(w.type(torch.FloatTensor)), sample)
        grad_value = quant_s(grad_temp.type(torch.FloatTensor))

    xi_v, xi_x = brown(u, gamma, eta, d)
    w_v_temp = w_v * math.exp(-gamma * eta) - u * (1 - math.exp(-gamma * eta)) / gamma * grad_value + xi_v
    w = w + (1 - math.exp(-gamma * eta)) / gamma * w_v + u * (gamma * eta + math.exp(-gamma * eta) - 1) / (
                gamma ** 2) * grad_value + xi_x
    w_v = w_v_temp

    return w, w_v


def Q_vc(mu, var):
    if var > VAR_F:
        x = mu + (var-VAR_F)**0.5*torch.randn(1)
        quant_x = quant_n(x)
        residual = x-quant_x
        # if torch.abs(residual) > D/2:
        #     print("X {}, Residual {}".format(x, residual))
        theta = quant_x+torch.sign(residual)*sample(torch.abs(residual), VAR_F)
    else:
        quant_mu = quant_s(mu)
        residual = mu-quant_mu
        p1 = torch.abs(residual) / D
        var_s = (1. - p1) * residual ** 2 + p1 * (-residual + torch.sign(residual) * D) ** 2
        if var > var_s:
            theta = quant_mu + sample(0, var - var_s)
        else:
            theta = quant_mu
    theta = torch.clamp(theta, min=-2**(WL-FL-1), max=2**(WL-FL-1)-2**(-FL))
    return theta

def sample(mu, var):
    p1 = (var+mu**2+mu*D)/(2*D**2)
    p2 = (var+mu**2-mu*D)/(2*D**2)
    u = torch.rand(1)
    # print("mu {}, var {}, p1{}".format(mu, var, p1))
    # if not (p1<=1 and p1>=0):
    #     pass
    # assert p1<=1 and p1>=0, "Invalid P1"
    # assert p2<=1 and p2>=0, "Invalid P2"
    if u < p1:
        return D
    elif u < p1+p2:
        return -D
    else:
        return 0


def sample_data(n):
    n1 = int(n/3)
    n2 = int(n/3*2)
    sample1 = np.random.normal(loc=-1.5, scale=0.5, size=n1)
    sample2 = np.random.normal(loc=1.5, scale=0.5, size=n2)

    sample = np.append(sample1, sample2)
    np.random.shuffle(sample)
    sample = torch.tensor(sample)

    return sample


# def grad_func(theta, d, sigma):
#     pdf_1 = torch.tensor(norm.pdf(theta, loc=-1, scale=0.6))
#     pdf_2 = torch.tensor(norm.pdf(theta, loc=1, scale=0.6))
#
#     temp = 1 / (1/2 * pdf_1 + 1/2 * pdf_2) * (1/2*pdf_1 * (theta + 1) * 4 + 1/2*pdf_2 * (theta - 1 )*4)
#
#     return temp + sigma * torch.rand((1, d))
def grad_func(theta, X):
    temp = torch.mean(4*(theta-X) + 8*X /(1+torch.exp(8*theta*X)))

    return temp


def funcU(x):
    pdf_1 = torch.tensor(norm.pdf(x, loc=-1.5, scale=0.5))
    pdf_2 = torch.tensor(norm.pdf(x, loc=1.5, scale=0.5))

    return 1/3 * pdf_1 + 2/3 * pdf_2


def densityplots_sgld(theta_list, stepsize):
    xStep = 0.01
    fig = plt.figure()
    sns.kdeplot(data=theta_list, color="y", label="{}_{}".format(args.dynamic, args.precision))

    x_axis = np.array(np.arange(-5, 5, xStep))
    y = funcU(x_axis)
    # y = y/sum(y)/xStep

    # [yhmc, xhmc] = np.histogram(theta_list.numpy(), x_axis)
    # yhmc = 1.0 * yhmc / sum(yhmc) / xStep
    # theta_list = theta_list.numpy()/ sum(theta_list.numpy()) / xStep

    # plt.plot(x_axis, theta_list, color="y", label="{}_{}".format(args.dynamic, args.precision))
    plt.plot(x_axis, y, label="True", lw=2, color='k')

    plt.legend(fontsize=10)
    plt.ylabel('Density', fontsize=13)
    plt.xlabel(stepsize, fontsize=13)

    return fig


def MCMC_theta_update(w, u, gamma, eta, d, w_v, batch, dynamic, precision="low_full"):
    if dynamic == 'hmc':
        if precision == "full":
            grad_value = grad_func(w, batch)  # quant_s(grad(quant_s(theta)))
            xi_v, xi_x = brown(u, gamma, eta, d)
            w_v_temp = w_v * math.exp(-gamma * eta) - u * (1 - math.exp(-gamma * eta)) / gamma * grad_value + xi_v
            w = w + (1 - math.exp(-gamma * eta)) / gamma * w_v + u * (gamma * eta + math.exp(-gamma * eta) - 1) / (
                    gamma ** 2) * grad_value + xi_x
            w_v = w_v_temp

        elif precision == "low_full":
            grad_temp = grad_func(quant_s(w.type(torch.FloatTensor)), batch)
            grad_value = quant_s(grad_temp.type(torch.FloatTensor))

            xi_v, xi_x = brown(u, gamma, eta, d)
            w_v_temp = w_v * math.exp(-gamma * eta) - u * (1 - math.exp(-gamma * eta)) / gamma * grad_value + xi_v
            w = w + (1 - math.exp(-gamma * eta)) / gamma * w_v + u * (gamma * eta + math.exp(-gamma * eta) - 1) / (
                    gamma ** 2) * grad_value + xi_x
            w_v = w_v_temp
        elif precision == "low_low":
            grad_value = quant_s(grad_func(w, batch).type(torch.FloatTensor))

            xi_v, xi_x = brown(u, gamma, eta, d)
            w_v_temp = quant_s(
                (w_v * math.exp(-gamma * eta) - u * (1 - math.exp(-gamma * eta)) / gamma * grad_value + xi_v).type(
                    torch.FloatTensor))
            w = quant_s(
                (w + (1 - math.exp(-gamma * eta)) / gamma * w_v + u * (gamma * eta + math.exp(-gamma * eta) - 1) / (
                        gamma ** 2) * grad_value + xi_x).type(torch.FloatTensor))
            w_v = w_v_temp
        elif precision == "vc":
            grad_value = quant_s(grad_func(w, batch).type(torch.FloatTensor))
            # xi_v, xi_x = brown(u, gamma, eta, d)
            var_v = u * (1 - math.exp(-2 * gamma * eta))
            var_x = (u / gamma ** 2) * (2 * gamma * eta + 4 * math.exp(-gamma * eta) - math.exp(-2 * gamma * eta) - 3)
            w_v_temp = Q_vc((w_v * math.exp(-gamma * eta) - u * (1 - math.exp(-gamma * eta)) / gamma * grad_value).type(
                torch.FloatTensor), var_v)
            w = Q_vc(
                (w + (1 - math.exp(-gamma * eta)) / gamma * w_v + u * (gamma * eta + math.exp(-gamma * eta) - 1) / (
                        gamma ** 2) * grad_value).type(torch.FloatTensor), var_x)
            w_v = w_v_temp
    elif dynamic == 'sgld':

        if precision == 'vc':
            grad_value = quant_s(grad_func(w, batch).type(torch.FloatTensor))
            # alpha = u * (gamma * eta + math.exp(-gamma * eta) - 1) / (
            #            gamma ** 2)
            alpha = eta
            mu_x = w - alpha * grad_value
            var = 2 * alpha
            w = Q_vc_sgld(mu_x, var)
            w_v = torch.zeros_like(w)
        elif precision == 'full':
            grad_value = grad_func(w, batch).type(torch.FloatTensor)
            # alpha = u * (gamma * eta + math.exp(-gamma * eta) - 1) / (
            #             gamma ** 2)
            alpha = eta
            mu_x = w - alpha * grad_value
            var = 2 * alpha
            w = mu_x + var ** .5 * torch.randn(d)
            w_v = torch.zeros_like(w)

        elif precision == 'low_low':
            grad_value = quant_s(grad_func(w, batch).type(torch.FloatTensor))
            # alpha = u * (gamma * eta + math.exp(-gamma * eta) - 1) / (
            #             gamma ** 2)
            alpha = eta
            mu_x = w - alpha * grad_value
            var = 2 * alpha
            w = quant_s(mu_x + var ** .5 * torch.randn(d))
            w_v = torch.zeros_like(w)

        elif precision == 'low_full':
            grad_value = quant_s(grad_func(quant_s(w.type(torch.FloatTensor)), batch).type(torch.FloatTensor))

            alpha = eta
            mu_x = w - alpha * grad_value
            var = 2 * alpha
            w = mu_x + var ** .5 * torch.randn(d)
            w_v = torch.zeros_like(w)

    return w, w_v


def MCMC(u, gamma, eta, d, sigma, dynamic, N, batch_size, precision="low_low", iteration=10000):
    theta_list = torch.zeros(iteration, d)
    vel_list = torch.zeros(iteration, d)
    theta = 1.5*torch.ones(1, d)
    vel = torch.zeros(1, d)
    data = sample_data(N)
    # sns.kdeplot(data)
    # x_axis = np.array(np.arange(-3, 3, 0.01))
    # y = funcU(x_axis)
    # plt.plot(x_axis, y, label='True')
    # plt.legend()
    # plt.show()
    batch = torch.tensor(np.random.choice(data, batch_size, replace=False))


    for i in range(iteration):
        theta_list[i], vel_list[i] = MCMC_theta_update(w=theta, u=u, gamma=gamma, eta=eta, d=d, w_v=vel, batch=batch,
                                               dynamic=dynamic, precision=precision)
        theta = theta_list[i]
        vel = vel_list[i]

        # theta_list[i], vel_list[i] = HMC_update(w=theta, u=u, gamma=gamma, eta=eta, d=d, sigma=sigma, nstep=nstep, precision=precision)
        # theta = theta_list[i]
        # vel = vel_list[i]

    return theta_list, vel_list


WL = args.WL
FL = args.FL
ITER = args.nsample
D = 1./(2**FL)
VAR_F = D**2/4
number = FixedPoint(wl=WL, fl=FL)
quant_s = quantizer(
    forward_number=number, forward_rounding="stochastic"
)
quant_n = quantizer(
    forward_number=number, forward_rounding="nearest"
)
U = args.U
Gamma = args.gamma
Eta = args.lr
Dim = args.dimension
Precision=args.precision
Nstep=args.nstep
data_size = 6000
batch_size= args.batch_size

path = '/home/wang4538/lowPrecisision/gaussian'

sample_list, vel_list = MCMC(U, Gamma, Eta, Dim, sigma=args.sigma, dynamic=args.dynamic, N=data_size, batch_size=batch_size, precision=Precision, iteration=ITER)
# data = {'{}_{}'.format(args.dynamic, args.precision): sample_list.numpy()}
df = pd.DataFrame(sample_list.numpy())
df.to_csv(os.path.join(path, 'result', '{}_{}_{}_u{}r{}_FL{}_{}_normal.csv'.format(args.dynamic, args.precision, args.lr, args.U, args.gamma, args.FL, args.type)))
fig = densityplots_sgld(sample_list, Eta)
fig.savefig(os.path.join(path, 'figs', '{}_{}_{}_u{}r{}_FL{}_{}_normal.png'.format(args.dynamic, args.precision, args.lr, args.U, args.gamma, args.FL, args.type)))




# for i in range(stepsize_lst.shape[0]):
#     plot = densityplots_sgld(full_list[i], low_full_list[i], low_low_list[i],
#                              vc_list[i], stepsize_lst[i])
#     pp = PdfPages('figs/feb_22/gaussian_mix_vc%s.pdf'%(stepsize_lst[i]))
#     pp.savefig(plot)
#     pp.close()
