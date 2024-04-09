import torch
from torch.optim import Optimizer, SGD, Adam
from qtorch import BlockFloatingPoint, FixedPoint, FloatingPoint
from qtorch.quant import *
import numpy as np

__all__ = ["OptimLP"]


def brown(u, gamma: torch.Tensor, eta: torch.Tensor, data: torch.Tensor, temperature, datasize):
    xi = torch.randn_like(data)
    var1 = u * (1 - torch.exp(-2 * gamma * eta)) / datasize * temperature
    var2 = (u / gamma ** 2) * (
            2 * gamma * eta + 4 * torch.exp(-gamma * eta) - torch.exp(-2 * gamma * eta) - 3) / datasize * temperature
    corr = u / gamma * (1 - 2 * torch.exp(-gamma * eta) + torch.exp(-2 * gamma * eta)) / datasize * temperature
    xi_v = xi * torch.sqrt(var1)
    xi_x = corr / var1 * xi_v + torch.sqrt(torch.max(var2 - corr ** 2 / var1, torch.tensor(.0))) * torch.randn_like(
        data)

    return xi_v, xi_x

class OptimLP(Optimizer):
    """
    A low-precision optimizer wrapper that handles weight, gradient, accumulator quantization.

    Args:
        - :attr: `optim`: underlying optimizer to use
        - :attr: `weight_quant`: a weight quantization function which takes a pytorch tensor and returns a tensor. If None, does not quantize weight.
        - :attr: `grad_quant`: a gradient quantization function which takes a pytorch tensor and returns a tensor. If None, does not quantize weight.
        - :attr: `grad_scaling`: float, scaling factor before apply gradient quantization.
        - :attr: `momentum_quant`: a momentum quantization function which takes a pytorch tensor and returns a tensor.
                                   If None, does not quantize weight.
        - :attr: `acc_quant`: a accumulator quantization function which takes
                              a pytorch tensor and returns a tensor. If not None, a
                              OptimLP object would create memory copies of model parameters that serve as
                              gradient accumulators. If None, does not use gradient accumulators.

    Example:
        >>> weight_q = quantizer(...) # define weight quantization
        >>> optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer = OptimLP(optiimizer, weight_quant=weight_q)
    """

    def __init__(
            self,
            optim,
            weight_quant=None,
            grad_scaling=1.0,
            grad_quant=None,
            momentum_quant=None,
            acc_quant=None,
            noise=False,
            temperature=1.0,
            datasize=None,
            WL=8,
            FL=8,
            EXP=8,
            MAN=7,
            quant_type='naive',
            number_type='fixed',
            vel_scaling=1.0,
            inverse_mass=1,
            friction=2,
            scale_x=False,
            dynamic='hmc'

    ):
        assert isinstance(optim, SGD) or isinstance(optim, Adam)
        super(OptimLP, self).__init__(
            optim.param_groups, optim.defaults
        )  # place holder

        # python dictionary does not copy by default
        self.param_groups = optim.param_groups
        self.optim = optim

        assert grad_scaling > 0, "gradient scaling must be positive"
        self.grad_scaling = grad_scaling

        self.weight_quant = weight_quant
        self.grad_quant = grad_quant
        self.momentum_quant = momentum_quant
        self.acc_quant = acc_quant
        self.dynamic = dynamic

        if isinstance(self.optim, SGD):
            self.momentum_keys = ["momentum_buffer"]
        elif isinstance(self.optim, Adam):
            # TODO: support amsgrad
            self.momentum_keys = ["exp_avg", "exp_avg_sq"]
        else:
            raise NotImplementedError("Only supporting Adam and SGD for now. ")
        #
        if self.acc_quant != None:
            self.weight_acc = {}
            for group in self.param_groups:
                for p in group["params"]:
                    self.weight_acc[p] = p.detach().clone()
        self.noise = noise
        self.temperature = temperature
        self.datasize = datasize
        self.quant_type = quant_type
        self.WL = WL
        self.FL = FL
        self.VEL_FL = FL
        self.EXP = EXP
        self.MAN = MAN
        self.number_type = number_type
        self.ebit = 8
        self.D = 1. / (2 ** FL)
        self.VEL_D = 1./ (2 ** FL)
        self.var_fix = self.D ** 2 / 4
        self.vel_var_fix = self.VEL_D ** 2 / 4
        if self.number_type == 'fixed':
            number = FixedPoint(wl=WL, fl=FL)
        elif self.number_type == 'block':
            number = BlockFloatingPoint(WL, dim=0)
        elif self.number_type == 'float':
            number = FloatingPoint(EXP, MAN)

        self.quant_s = quantizer(
                forward_number=number, forward_rounding="stochastic"
        )
        self.quant_n = quantizer(
            forward_number=number, forward_rounding="nearest"
        )

        #


        self.vel_scaling_constant = vel_scaling
        self.scale_param = scale_x
        self.vel_scaling_parameter = {}
        self.velocity = {}
        self.velocity_acc = {}
        self.param_scale = {}
        self.param_scaled = {}
        for group in self.param_groups:
            for p in group['params']:
                self.param_scaled[p] = p.detach().clone()
                self.velocity[p] = torch.zeros_like(p.data)
                self.velocity_acc[p] = torch.zeros_like(p.data)
                self.vel_scaling_parameter[p] = 1
                self.param_scale[p] = 1

        self.inverse_mass = torch.tensor(inverse_mass, device=self.velocity[p].device,
                                         dtype=self.velocity[p].dtype)
        self.friction = torch.tensor(friction, device=self.velocity[p].device, dtype=self.velocity[p].dtype)

    def step(self, closure=None):
        """
        Performs one step of optimization with the underlying optimizer.
        Quantizes gradient and momentum before stepping. Quantizes gradient accumulator and weight after stepping.
        """
        loss = None
        # quantize gradient
        if not self.grad_quant is None and self.quant_type != 'full':
            for group in self.param_groups:
                for p in group["params"]:
                    p.grad.data = self.grad_quant(p.grad.data * self.grad_scaling)

        # switch acc into weight before stepping
        if not self.acc_quant is None:
            for group in self.param_groups:
                for p in group["params"]:
                    p.data = self.weight_acc[p].data

        self.update_params()
        # self.optim.step()

        # switch weight into acc after stepping and quantize
        if not self.acc_quant is None:
            for group in self.param_groups:
                for p in group["params"]:
                    if self.acc_quant == "full":
                        self.weight_acc[p].data = p.data
                    else:
                        p.data = self.weight_acc[p].data = self.acc_quant(p.data).data

            for p in self.velocity:
                if self.acc_quant == 'full':
                    self.velocity_acc[p].data = self.velocity[p].data
                else:
                    self.velocity[p].data = self.velocity_acc[p].data = self.weight_quant(self.velocity[p].data).data

        # quantize weight from acc
        if (not self.weight_quant is None) and (self.quant_type == 'naive'):
            for group in self.param_groups:
                for p in group["params"]:
                    p.data = self.weight_quant(p.data).data
            for p in self.velocity:
                self.velocity[p].data = self.weight_quant(self.velocity[p].data).data

        if self.scale_param == 'true':
            self.param_rescale()

        return loss

    # @torch.no_grad()
    def update_params(self):
        for group, velocity in zip(self.param_groups, self.velocity):
            weight_decay = group['weight_decay']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p, vel in zip(group['params'], velocity):
                param_state = self.optim.state[p]
                d_p = p.grad.data
                if self.noise:
                    if self.dynamic == 'sgld':
                        d_p.add_(p.data, alpha=weight_decay)
                        p.data.add_(d_p, alpha=-group['lr'])
                        if self.quant_type == 'vc':
                            var = 2.0 * group['lr'] * self.temperature / self.datasize
                            if self.number_type == 'block':
                                self.FL = self.compute_fl(p.data)
                                self.D = 1. / (2 ** self.FL)
                                self.var_fix = self.D ** 2 / 4.
                                p.data = self.fp_Q_vc(p.data, var)
                            elif self.number_type == 'float':
                                self.FL = self.compute_fl_float(p.data)
                                self.D = 1. / (2 ** self.FL)
                                self.var_fix = self.D ** 2 / 4.
                                p.data = self.fp_Q_vc(p.data, var)
                            else:
                                p.data = self.Q_vc(p.data, var)
                        else:
                            var = 2.0 * group['lr'] * self.temperature / self.datasize
                            eps = torch.randn(p.size(), device='cuda')
                            noise = var ** .5 * eps
                            p.data.add_(noise)
                    elif self.dynamic == 'hmc2':

                        if not self.acc_quant is None:
                            vel = self.velocity[p]
                        else:
                            vel = self.velocity[p] * self.vel_scaling_parameter[p]

                        self.velocity[p] = vel * torch.exp(-self.friction * group['lr']) - self.inverse_mass * (
                                1 - torch.exp(-self.friction * group['lr'])) / self.friction * d_p
                        w = (1 - torch.exp(-self.friction * group['lr'])) / self.friction * vel + \
                            self.inverse_mass * (self.friction * group['lr'] + torch.exp(
                            -self.friction * group['lr']) - 1) / (self.friction ** 2) * d_p
                        var_x = (self.inverse_mass / self.friction ** 2) * (
                                    2 * self.friction * group['lr'] / self.datasize
                                    + 4 * torch.exp(-self.friction * group['lr'] / self.datasize)
                                    - torch.exp(
                                -2 * self.friction * group['lr'] / self.datasize) - 3) * self.temperature
                        var_v = self.inverse_mass * (1 - torch.exp(
                            -2 * self.friction * group['lr'] / self.datasize)) * self.temperature
                        p.data.add_(w)
                        if self.acc_quant is None and self.number_type == 'fixed':
                            self.update_vel_scaling(p)
                        if self.scale_param == 'true':
                            self.update_param_scaling(p)
                        if self.noise:
                            if self.quant_type == 'vc':
                                if self.scale_param == 'true':
                                    p.data = self.Q_vc(p.data / self.param_scale[p],
                                                       var_x / (self.param_scale[p] ** 2))
                                else:
                                    p.data = self.Q_vc(p.data, var_x)


                                self.velocity[p].data = self.Q_vc(
                                    self.velocity[p].data / self.vel_scaling_parameter[p],
                                    var_v / (self.vel_scaling_parameter[p] ** 2))

                            else:
                                xi_v, xi_x = brown(self.inverse_mass, self.friction, group['lr'], d_p, self.temperature,
                                                   self.datasize)
                                if not self.acc_quant is None:
                                    self.velocity[p].data = self.velocity[p].data + xi_v
                                elif self.acc_quant is None:
                                    self.velocity[p].data = (self.velocity[p].data + xi_v) / self.vel_scaling_parameter[p]
                                
                                if self.scale_param == 'true':
                                    p.data = p.data / self.param_scale[p]
                                    p.data.add_(xi_x / self.param_scale[p])
                                else:
                                    p.data.add_(xi_x)

    def quant_n_hf(self, mu):
        return mu.half()

    def compute_fl_float(self, mu):
        max_entry = torch.abs(mu)
        max_exponent = torch.floor(torch.log2(max_entry))
        max_exponent = torch.clamp(max_exponent, -2 ** (self.EXP - 1), 2 ** (self.EXP - 1) - 1)
        return self.MAN - max_exponent

    def compute_fl(self, mu):
        max_entry = torch.max(torch.abs(mu.view(mu.size(0), -1)), 1)[0]
        max_exponent = torch.floor(torch.log2(max_entry))
        max_exponent = torch.clamp(max_exponent, -2 ** (self.ebit - 1), 2 ** (self.ebit - 1) - 1)
        max_exponent = max_exponent.view([mu.size(0)] + [1 for _ in range(mu.dim() - 1)])
        max_exponent = max_exponent.expand([-1] + [mu.size(i) for i in range(1, mu.dim())])
        return self.WL - 2 - max_exponent

    def Q_vc(self, mu, var):
        if var > self.var_fix:
            x = mu + (var - self.var_fix) ** .5 * torch.randn(mu.size(), device='cuda')
            quant_x = self.quant_n(x)
            residual = x - quant_x
            theta = quant_x + torch.sign(residual) * self.sample_mu(torch.abs(residual))
        else:
            quant_mu = self.quant_s(mu)
            residual = mu - quant_mu
            p1 = torch.abs(residual) / self.D
            var_s = (1. - p1) * residual ** 2 + p1 * (-residual + torch.sign(residual) * self.D) ** 2
            v = var - var_s
            v[(v < 0).nonzero(as_tuple=True)] = 0
            theta = quant_mu + self.sample(v)

        theta = torch.clamp(theta, min=-2 ** (self.WL - self.FL - 1),
                            max=2 ** (self.WL - self.FL - 1) - 2 ** (-self.FL))
        return theta

    
    def fp_Q_vel_vc(self, mu, var):
        FL0 = self.VEL_FL.detach()
        var = torch.zeros(1, device='cuda') + var
        var = var.expand(self.vel_var_fix.size())
        ind = (var <= self.vel_var_fix).nonzero(as_tuple=True)

        x = mu + torch.max(torch.tensor(0), (var - self.vel_var_fix)) ** .5 * torch.randn(mu.size(), device='cuda')
        quant_x = self.quant_n(x)
        if self.number_type == 'block':
            self.VEL_FL = self.compute_fl(x)
            self.VEL_D = 1. / (2 ** self.VEL_FL)
        elif self.number_type == 'float':
            self.VEL_FL = self.compute_fl_float(x)
            self.VEL_D = 1. / (2 ** self.VEL_FL)
        residual = x - quant_x
        theta = quant_x + torch.sign(residual) * self.fp_vel_sample_mu(torch.abs(residual))

        quant_mu = self.quant_s(mu)
        residual = mu - quant_mu
        p1 = torch.abs(residual) / self.VEL_D
        var_s = (1. - p1) * residual ** 2 + p1 * (-residual + torch.sign(residual) * self.VEL_D) ** 2
        v = var - var_s
        v[(v < 0).nonzero(as_tuple=True)] = 0
        theta1 = quant_mu + self.fp_vel_sample(v)

        theta[ind] = theta1[ind]
        self.VEL_FL[ind] = FL0[ind]
        if self.number_type == 'float':
            pass
        else:
            theta = torch.clamp(theta, min=-2 ** (self.WL - self.VEL_FL - 1),
                                max=2 ** (self.WL - self.VEL_FL - 1) - 2 ** (-self.VEL_FL))
        return theta

    def fp_Q_vc(self, mu, var):
        FL0 = self.FL.detach()
        var = torch.zeros(1, device='cuda') + var
        var = var.expand(self.var_fix.size())
        ind = (var <= self.var_fix).nonzero(as_tuple=True)

        x = mu + (var - self.var_fix) ** .5 * torch.randn(mu.size(), device='cuda')
        quant_x = self.quant_n(x)
        if self.number_type == 'block':
            self.FL = self.compute_fl(x)
            self.D = 1. / (2 ** self.FL)
        elif self.number_type == 'float':
            self.FL = self.compute_fl_float(x)
            self.D = 1. / (2 ** self.FL)
        residual = x - quant_x
        theta = quant_x + torch.sign(residual) * self.fp_sample_mu(torch.abs(residual))

        quant_mu = self.quant_s(mu)
        residual = mu - quant_mu
        p1 = torch.abs(residual) / self.D
        var_s = (1. - p1) * residual ** 2 + p1 * (-residual + torch.sign(residual) * self.D) ** 2
        v = var - var_s
        v[(v < 0).nonzero(as_tuple=True)] = 0
        theta1 = quant_mu + self.fp_sample(v)

        theta[ind] = theta1[ind]
        self.FL[ind] = FL0[ind]
        if self.number_type == 'float':
            pass
        else:
            theta = torch.clamp(theta, min=-2 ** (self.WL - self.FL - 1),
                                max=2 ** (self.WL - self.FL - 1) - 2 ** (-self.FL))
        return theta

    def sample(self, var):
        p1 = var / (2 * self.D ** 2)
        u = torch.rand(var.size(), device='cuda')
        s = torch.zeros(var.size(), device='cuda')
        s[(u < p1).nonzero(as_tuple=True)] = self.D
        u[(u < p1).nonzero(as_tuple=True)] = 10.
        s[(u < (2 * p1)).nonzero(as_tuple=True)] = -self.D
        return s

    def sample_mu(self, mu):
        p1 = (self.var_fix + mu ** 2 + mu * self.D) / (2 * self.D ** 2)
        p2 = (self.var_fix + mu ** 2 - mu * self.D) / (2 * self.D ** 2)
        u = torch.rand(mu.size(), device='cuda')
        s = torch.zeros(mu.size(), device='cuda')
        s[(u < p1).nonzero(as_tuple=True)] = self.D
        u[(u < p1).nonzero(as_tuple=True)] = 10.
        s[(u < (p1 + p2)).nonzero(as_tuple=True)] = -self.D
        return s

    def _sample(self, mu, var):
        p1 = (var + mu ** 2 + mu * self.D) / (2 * self.D ** 2)
        p2 = (var + mu ** 2 - mu * self.D) / (2 * self.D ** 2)
        u = torch.rand(mu.size(), device='cuda')
        s = torch.zeros(mu.size(), device='cuda')
        s[(u < p1).nonzero(as_tuple=True)] = self.D
        u[(u < p1).nonzero(as_tuple=True)] = 10
        s[(u < (p1 + p2)).nonzero(as_tuple=True)] = -self.D
        return s

    def fp_sample(self, var):
        p1 = var / (2 * self.D ** 2)
        u = torch.rand(var.size(), device='cuda')
        s = torch.zeros(var.size(), device='cuda')
        ind1 = (u < p1).nonzero(as_tuple=True)
        s[ind1] = self.D[ind1]
        u[ind1] = 10.
        ind2 = (u < (2 * p1)).nonzero(as_tuple=True)
        s[ind2] = -self.D[ind2]
        return s

    def fp_vel_sample(self, var):
        p1 = var / (2 * self.VEL_D ** 2)
        u = torch.rand(var.size(), device='cuda')
        s = torch.zeros(var.size(), device='cuda')
        ind1 = (u < p1).nonzero(as_tuple=True)
        s[ind1] = self.VEL_D[ind1]
        u[ind1] = 10.
        ind2 = (u < (2 * p1)).nonzero(as_tuple=True)
        s[ind2] = -self.VEL_D[ind2]
        return s

    def fp_sample_mu(self, mu):
        p1 = (self.var_fix + mu ** 2 + mu * self.D) / (2 * self.D ** 2)
        p2 = (self.var_fix + mu ** 2 - mu * self.D) / (2 * self.D ** 2)
        u = torch.rand(mu.size(), device='cuda')
        s = torch.zeros(mu.size(), device='cuda')
        ind1 = (u < p1).nonzero(as_tuple=True)
        s[ind1] = self.D[ind1]
        u[ind1] = 10.
        ind2 = (u < (p1 + p2)).nonzero(as_tuple=True)
        s[ind2] = -self.D[ind2]
        return s

    def fp_vel_sample_mu(self, mu):
        p1 = (self.vel_var_fix + mu ** 2 + mu * self.VEL_D) / (2 * self.VEL_D ** 2)
        p2 = (self.vel_var_fix + mu ** 2 - mu * self.VEL_D) / (2 * self.VEL_D ** 2)
        u = torch.rand(mu.size(), device='cuda')
        s = torch.zeros(mu.size(), device='cuda')
        ind1 = (u < p1).nonzero(as_tuple=True)
        s[ind1] = self.VEL_D[ind1]
        u[ind1] = 10.
        ind2 = (u < (p1 + p2)).nonzero(as_tuple=True)
        s[ind2] = -self.VEL_D[ind2]
        return s  

    def update_vel_scaling(self, p):
        max = 2 ** (self.WL-self.FL-1) - 2 ** (-self.FL)
        scale = torch.max(torch.abs(self.velocity[p].data))/(self.vel_scaling_constant * max)
        self.vel_scaling_parameter[p] = torch.min(torch.tensor(1), scale)

    def param_rescale(self):
        for group in self.param_groups:
            for p in group['params']:
                p.data = p.data * self.param_scale[p]
                # self.param_scaled[p].data = self.param_scaled[p].data * self.param_scale[p]

    def update_param_scaling(self, p):
        max = 2 ** (self.WL - self.FL - 1) - 2 ** (-self.FL)
        scale = torch.max(torch.abs(p.data))/(self.vel_scaling_constant*max)
        self.param_scale[p] = torch.min(torch.tensor(1), scale)



    def __repr__(self):
        return "LP Optimizer: {}".format(self.optim.__repr__())

    def __str__(self):
        return "LP Optimizer: {}".format(self.optim.__str__())
