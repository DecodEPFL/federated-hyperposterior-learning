import torch, gpytorch, math, os, sys, warnings
import numpy as np
from collections import OrderedDict
from pyro.distributions import Normal

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from config import device
from server.util import find_root_by_bounding, warning_show, warning_format

#warnings.showwarning = warning_show
warnings.formatwarning = warning_format

# change gpytorch settings to avoid non-PSD errors
gpytorch.settings.cholesky_jitter(1e-4, 1e-6)
# Default for float: 1e-6, Default for double: 1e-8
gpytorch.settings.cholesky_max_tries(value=8) # default = 3

""" ----------------------------------------------------"""
""" ------------ Probability Distributions ------------ """
""" ----------------------------------------------------"""

from torch.distributions import Distribution
from torch.distributions import TransformedDistribution, AffineTransform

class AffineTransformedDistribution(TransformedDistribution):
    r"""
    Implements an affine transformation of a probability distribution p(x)

    x_transformed = mean + std * x , x \sim p(x)

    Args:
        base_dist: (torch.distributions.Distribution) probability distribution to transform
        normalization_mean: (np.ndarray) additive factor to add to x
        normalization_std: (np.ndarray) multiplicative factor for scaling x
    """

    def __init__(self, base_dist, normalization_mean, normalization_std):
        self.loc_tensor = torch.tensor(normalization_mean).float().reshape((1,)).to(device)
        self.scale_tensor = torch.tensor(normalization_std).float().reshape((1,)).to(device)
        normalization_transform = AffineTransform(loc=self.loc_tensor, scale=self.scale_tensor)
        super().__init__(base_dist, normalization_transform)

    @property
    def mean(self):
        return self.transforms[0](self.base_dist.mean)
    @mean.setter
    def set_mean(self, value):
        self.mean = value

    @property
    def stddev(self):
        return torch.exp(torch.log(self.base_dist.stddev) + torch.log(self.scale_tensor))
    @stddev.setter
    def set_stddev(self, value):
        self.stddev = value

    @property
    def variance(self):
        return torch.exp(torch.log(self.base_dist.variance) + 2 * torch.log(self.scale_tensor))

class UnnormalizedExpDist(Distribution):
    r"""
    Creates a an unnormalized distribution with density function with
    density proportional to exp(exponent_fn(value))

    Args:
      exponent_fn: callable that outputs the exponent
    """

    def __init__(self, exponent_fn):
        self.exponent_fn = exponent_fn
        super().__init__()

    @property
    def arg_constraints(self):
        return {}

    def log_prob(self, value):
        return self.exponent_fn(value)

class FactorizedNormal(Distribution):

    def __init__(self, loc, scale, summation_axis=-1):
        self.normal_dist = torch.distributions.Normal(loc, scale)
        self.summation_axis = summation_axis

    def log_prob(self, value):
        return torch.sum(self.normal_dist.log_prob(value), dim=self.summation_axis)

class EqualWeightedMixtureDist(Distribution):

    def __init__(self, dists, batched=False, num_dists=None, comb_weights=None):
        self.batched = batched
        if batched:
            assert isinstance(dists, torch.distributions.Distribution)
            self.num_dists = dists.batch_shape if num_dists is None else num_dists
            event_shape = dists.event_shape
            self.in_dim = dists.mean.shape[1]
        else:
            assert all([isinstance(d, torch.distributions.Distribution) for d in dists])
            event_shape = dists[0].event_shape
            self.num_dists = len(dists)
            self.in_dim = dists[0].mean.flatten().shape[0]
            assert all([dist.mean.flatten().shape[0]==self.in_dim for dist in dists])
        self.dists = dists

        super().__init__(event_shape=event_shape)

        n = self.num_dists
        if isinstance(n, torch.Size):
            n = torch.tensor(n)[0]

        if comb_weights is None:
            print('[WARN] combination weights missing')
            comb_weights = torch.ones(n, self.in_dim).to(device)/n
        assert isinstance(comb_weights, torch.Tensor)
        if (len(comb_weights.shape)==2) and (comb_weights.shape[0]==n) and (comb_weights.shape[1]==self.in_dim):
            self.comb_weights = comb_weights
        elif (len(comb_weights.shape)==1) and (comb_weights.shape[0]==n):
            self.comb_weights = comb_weights.reshape(-1,1).repeat(1, self.in_dim)
        else:
            raise NotImplementedError


    @property
    def mean(self):
        if self.batched:
            means = self.dists.mean
        else:
            means = torch.stack([dist.mean for dist in self.dists], dim=0)
        # weight
        means = torch.multiply(
            means,
            self.comb_weights
        )
        return torch.sum(means, dim=0)

    @property
    def stddev(self):
        return torch.sqrt(self.variance)

    @property
    def variance(self):
        if self.batched:
            means = self.dists.mean
            vars = self.dists.variance
        else:
            means = torch.stack([dist.mean for dist in self.dists], dim=0)
            vars = torch.stack([dist.variance for dist in self.dists], dim=0)
        # means and vars are of torch.Size([num_particles, num_test_samples])

        var = torch.sum(
            torch.mul(
                self.comb_weights,
                (means - self.mean)**2 + vars
            ),
            dim=0
        )
        # varis of torch.Size([num_test_samples])
        return var

    @property
    def arg_constraints(self):
        return {}

    def log_prob(self, value):
        with gpytorch.settings.cholesky_jitter(1e-1): # increase jitter to avoid non-psd cov
            if self.batched:
                log_probs_dists = self.dists.log_prob(value)
            else:
                log_probs_dists = torch.stack([dist.log_prob(value) for dist in self.dists])
        return torch.logsumexp(log_probs_dists+torch.log(self.comb_weights[:, 0]), dim=0)


    def cdf(self, value):
        if self.batched:
            cum_p = self.dists.cdf(value)
        else:
            cum_p = torch.stack([dist.cdf(value) for dist in self.dists])
        assert cum_p.shape[0] == self.num_dists
        return torch.sum(torch.multiply(cum_p, self.comb_weights), dim=0)

    def icdf(self, quantile):
        left = - 1e8 * torch.ones(quantile.shape)
        right = + 1e8 * torch.ones(quantile.shape)
        fun = lambda x: self.cdf(x) - quantile
        return find_root_by_bounding(fun, left, right)



class CatDist(Distribution):

    def __init__(self, dists, reduce_event_dim=True):
        assert all([len(dist.event_shape) == 1 for dist in dists])
        assert all([len(dist.batch_shape) == 0 for dist in dists])
        self.reduce_event_dim = reduce_event_dim
        self.dists = dists
        self._event_shape = torch.Size((sum([dist.event_shape[0] for dist in self.dists]),))


    def sample(self, sample_shape=torch.Size()):
        return self._sample(sample_shape, sample_fn='sample')

    def rsample(self, sample_shape=torch.Size()):
        return self._sample(sample_shape, sample_fn='rsample')

    def log_prob(self, value):
        idx = 0
        log_probs = []
        for dist in self.dists:
            n = dist.event_shape[0]
            if value.ndim == 1:
                val = value[idx:idx+n]
            elif value.ndim == 2:
                val = value[:, idx:idx + n]
            elif value.ndim == 2:
                val = value[:, :, idx:idx + n]
            else:
                raise NotImplementedError('Can only handle values up to 3 dimensions')
            log_probs.append(dist.log_prob(val))
            idx += n

        for i in range(len(log_probs)):
            if log_probs[i].ndim == 0:
                log_probs[i] = log_probs[i].reshape((1,))

        if self.reduce_event_dim:
            return torch.sum(torch.stack(log_probs, dim=0), dim=0)
        return torch.stack(log_probs, dim=0)

    def _sample(self, sample_shape, sample_fn='sample'):
        return torch.cat([getattr(d, sample_fn)(sample_shape).to(device) for d in self.dists], dim=-1)

""" ----------------------------------------------------"""
""" ------------------ Neural Network ------------------"""
""" ----------------------------------------------------"""

class NeuralNetwork(torch.nn.Sequential):
    """Trainable neural network kernel function for GPs."""
    def __init__(
        self, input_dim=2, output_dim=2, layer_sizes=(64, 64),
        nonlinearity_hidden=torch.tanh, nonlinearity_output=torch.tanh,
        weight_norm=False, prefix='', requires_bias={'out':True, 'hidden':True},):
        super(NeuralNetwork, self).__init__()
        self.nonlinearity_hidden = nonlinearity_hidden
        self.nonlinearity_output = nonlinearity_output
        self.n_layers = len(layer_sizes)
        self.prefix = prefix

        if weight_norm:
            _normalize = torch.nn.utils.weight_norm
        else:
            _normalize = lambda x: x

        self.layers = []
        prev_size = input_dim
        for i, size in enumerate(layer_sizes):
            setattr(self, self.prefix + 'fc_%i'%(i+1), _normalize(
                torch.nn.Linear(prev_size, size, device=device, bias=requires_bias['hidden']))
                )
            prev_size = size
        setattr(self, self.prefix + 'out', _normalize(
            torch.nn.Linear(prev_size, output_dim, device=device, bias=requires_bias['out']))
            )

    def forward(self, x):
        output = x
        for i in range(1, self.n_layers+1):
            output = getattr(self, self.prefix + 'fc_%i'%i)(output)
            if not self.nonlinearity_hidden is None:
                output = self.nonlinearity_hidden(output)
        output = getattr(self, self.prefix + 'out')(output)
        if not self.nonlinearity_output is None:
            output = self.nonlinearity_output(output)
        return output

    def forward_parametrized(self, x, params):
        output = x
        param_idx = 0
        for i in range(1, self.n_layers + 1):
            output = F.linear(output, params[param_idx], params[param_idx+1])
            if not self.nonlinearity_hidden is None:
                output = self.nonlinearity_hidden(output)
            param_idx += 2
        output = F.linear(output, params[param_idx], params[param_idx+1])
        if not self.nonlinearity_output is None:
            output = self.nonlinearity_output(output)
        return output

""" ----------------------------------------------------"""
""" ------------ Vectorized Neural Network -------------"""
""" ----------------------------------------------------"""

import torch.nn as nn
import torch.nn.functional as F


class VectorizedModel:

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def parameter_shapes(self):
        raise NotImplementedError

    def named_parameters(self):
        raise NotImplementedError

    def parameters(self):
        return list(self.named_parameters().values())

    def set_parameter(self, name, value):
        if len(name.split('.')) == 1:
            setattr(self, name, value)
        else:
            remaining_name = ".".join(name.split('.')[1:])
            getattr(self, name.split('.')[0]).set_parameter(remaining_name, value)

    def set_parameters(self, param_dict):
        for name, value in param_dict.items():
            self.set_parameter(name, value)

    def parameters_as_vector(self):
        return torch.cat(self.parameters(), dim=-1)

    def set_parameters_as_vector(self, value):
        # value is reshaped to the parameter shape
        idx = 0
        for name, shape in self.parameter_shapes().items():
            idx_next = idx + shape[-1]
            if value.ndim == 1:
                self.set_parameter(name, value[idx:idx_next])
            elif value.ndim == 2:
                self.set_parameter(name, value[:, idx:idx_next])
            else:
                raise AssertionError
            idx = idx_next
        assert idx_next == value.shape[-1]





class LinearVectorized(VectorizedModel):
    def __init__(self, input_dim, output_dim, requires_bias=True, nonlinearity=torch.tanh):
        super().__init__(input_dim, output_dim)

        self.weight = torch.normal(0, 1, size=(input_dim * output_dim,), device=device, requires_grad=True, dtype=torch.double)
        self.weight_dist = None
        self.bias = torch.zeros(size=(output_dim,), device=device, requires_grad=requires_bias, dtype=torch.double)
        self.bias_dist = None
        self.bias.requires_grad=requires_bias
        self.nonlinearity=nonlinearity # nonlinearity which will be applied on top of this letter. affects initialization
        self.reset_parameters()  # initialize weights of NN hidden and output layer by Kaiming method


    def reset_parameters(self):
        # initialize weights using the Kaiming method and set weights_dist
        self.weight, self.weight_dist = _kaiming_uniform_batched(self.weight, fan=self.input_dim, a=math.sqrt(5), nonlinearity=self.nonlinearity)
        if self.bias is not None and self.bias.requires_grad:
            fan_in = self.output_dim
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            # bias is initialized from a uniform dist, but since log prob might be -inf, approx by normal
            self.bias_dist = Normal(torch.tensor([0]*self.bias.size(dim=0)).float(),
                                    torch.tensor([math.sqrt(bound*(bound+1)/3)]*self.bias.size(dim=0)).float())


    def forward(self, x):
        if self.weight.ndim == 2 or self.weight.ndim == 3:
            model_batch_size = self.weight.shape[0]

            # batched computation
            if self.weight.ndim == 3:
                assert self.weight.shape[-2] == 1 and self.bias.shape[-2] == 1
            if len(self.bias.shape)==1: # if bias is 1D
                self.bias = self.bias.repeat(self.weight.shape[0], 1)


            W = self.weight.view(model_batch_size, self.output_dim, self.input_dim)
            b = self.bias.view(model_batch_size, self.output_dim)

            if x.ndim == 2:
                # introduce new dimension 0
                x = torch.reshape(x, (1, x.shape[0], x.shape[1]))
                # tile dimension 0 to model_batch size
                x = x.repeat(model_batch_size, 1, 1)
            else:
                assert x.ndim == 3 and x.shape[0] == model_batch_size
            # out dimensions correspond to [nn_batch_size, data_batch_size, out_features)
            return torch.bmm(x.float(), W.float().permute(0, 2, 1)) + b[:, None, :].float()
        elif self.weight.ndim == 1:
            return F.linear(x, self.weight.view(self.output_dim, self.input_dim), self.bias)
        else:
            raise NotImplementedError

    def parameter_shapes(self):
        if self.bias.requires_grad:
            return OrderedDict(bias=self.bias.shape, weight=self.weight.shape)
        else:
            return OrderedDict(weight=self.weight.shape)

    def named_parameters(self):
        if self.bias.requires_grad:
            return OrderedDict(bias=self.bias, weight=self.weight)
        else:
            return OrderedDict(weight=self.weight)

    def __call__(self, *args, **kwargs):
        return self.forward( *args, **kwargs)

class NeuralNetworkVectorized(VectorizedModel):
    """Trainable neural network that batches multiple sets of parameters. That is, each
    """
    def __init__(self, input_dim, output_dim, layer_sizes=(64, 64),  nonlinearity_hidden=torch.tanh,
                 nonlinearity_output=torch.tanh, requires_bias={'out':True, 'hidden':True}):
        # requires_bias: add bias to the hidden and output layers or not
        super().__init__(input_dim, output_dim)

        self.nonlinearity_hidden = nonlinearity_hidden
        self.nonlinearity_output = nonlinearity_output
        self.n_layers = len(layer_sizes)
        prev_size = input_dim
        for i, size in enumerate(layer_sizes):
            setattr(
                self, 'fc_%i'%(i+1),
                LinearVectorized(
                    prev_size, size,
                    requires_bias=requires_bias['hidden'],
                    nonlinearity=self.nonlinearity_hidden))
            prev_size = size
        setattr(
            self, 'out',
            LinearVectorized(
                prev_size, output_dim,
                requires_bias=requires_bias['out'],
                nonlinearity=self.nonlinearity_output))


    def forward(self, x):
        output = x
        for i in range(1, self.n_layers + 1):
            output = getattr(self, 'fc_%i' % i)(output)
            if self.nonlinearity_hidden is not None:
                output = self.nonlinearity_hidden(output)
        output = getattr(self, 'out')(output)
        if self.nonlinearity_output is not None:
            output = self.nonlinearity_output(output)
        return output

    def parameter_shapes(self):
        param_dict = OrderedDict()

        # hidden layers
        for i in range(1, self.n_layers + 1):
            layer_name = 'fc_%i' % i
            for name, param in getattr(self, layer_name).parameter_shapes().items():
                param_dict[layer_name + '.' + name] = param

        # last layer
        layer_name = 'out'
        for name, param in getattr(self, layer_name).parameter_shapes().items():
            param_dict[layer_name + '.' + name] = param

        return param_dict

    def named_parameters(self):
        param_dict = OrderedDict()

        # hidden layers
        for i in range(1, self.n_layers + 1):
            layer_name = 'fc_%i' % i
            for name, param in getattr(self, layer_name).named_parameters().items():
                param_dict[layer_name + '.' + name] = param

        # last layer
        layer_name = 'out'
        for name, param in getattr(self, layer_name).named_parameters().items():
            param_dict[layer_name + '.' + name] = param

        return param_dict

    def __call__(self, *args, **kwargs):
        return self.forward( *args, **kwargs)

""" Initialization Helpers """

def _kaiming_uniform_batched(tensor, fan, a=0.0, nonlinearity=torch.tanh):
    nonlinearity='linear' if nonlinearity==None else nonlinearity.__name__
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound), Normal(torch.tensor([0]*tensor.size(dim=0)).float(),
                                                      torch.tensor([math.sqrt(bound*(bound+1)/3)]*tensor.size(dim=0)).float())


""" ----------------------------------------------------"""
""" ------------------ GP components -------------------"""
""" ----------------------------------------------------"""

from gpytorch.means import Mean
from gpytorch.kernels import Kernel
from gpytorch.functions import RBFCovariance
from gpytorch.module import Module
#from gpytorch.utils.broadcasting import _mul_broadcast_shape

# --- functions from gpytorch 1.6.0 ---
def _mul_broadcast_shape(*shapes, error_msg=None):
    """Compute dimension suggested by multiple tensor indices (supports broadcasting)"""

    # Pad each shape so they have the same number of dimensions
    num_dims = max(len(shape) for shape in shapes)
    shapes = tuple([1] * (num_dims - len(shape)) + list(shape) for shape in shapes)

    # Make sure that each dimension agrees in size
    final_size = []
    for size_by_dim in zip(*shapes):
        non_singleton_sizes = tuple(size for size in size_by_dim if size != 1)
        if len(non_singleton_sizes):
            if any(size != non_singleton_sizes[0] for size in non_singleton_sizes):
                if error_msg is None:
                    raise RuntimeError("Shapes are not broadcastable for mul operation")
                else:
                    raise RuntimeError(error_msg)
            final_size.append(non_singleton_sizes[0])
        # In this case - all dimensions are singleton sizes
        else:
            final_size.append(1)

    return torch.Size(final_size)


def _matmul_broadcast_shape(shape_a, shape_b, error_msg=None):
    """Compute dimension of matmul operation on shapes (supports broadcasting)"""
    m, n, p = shape_a[-2], shape_a[-1], shape_b[-1]

    if len(shape_b) == 1:
        if n != p:
            if error_msg is None:
                raise RuntimeError(f"Incompatible dimensions for matmul: {shape_a} and {shape_b}")
            else:
                raise RuntimeError(error_msg)
        return shape_a[:-1]

    if n != shape_b[-2]:
        if error_msg is None:
            raise RuntimeError(f"Incompatible dimensions for matmul: {shape_a} and {shape_b}")
        else:
            raise RuntimeError(error_msg)

    tail_shape = torch.Size([m, p])

    # Figure out batch shape
    batch_shape_a = shape_a[:-2]
    batch_shape_b = shape_b[:-2]
    if batch_shape_a == batch_shape_b:
        bc_shape = batch_shape_a
    else:
        bc_shape = _mul_broadcast_shape(batch_shape_a, batch_shape_b)
    return bc_shape + tail_shape





class ConstantMeanLight(gpytorch.means.Mean):
    def __init__(self, constant=torch.ones(1), batch_shape=torch.Size()):
        super(ConstantMeanLight, self).__init__()
        self.batch_shape = batch_shape
        self.constant = constant

    def forward(self, input):
        if input.shape[:-2] == self.batch_shape:
            return self.constant.expand(input.shape[:-1])
        else:
            return self.constant.expand(_mul_broadcast_shape(input.shape[:-1], self.constant.shape))


class ZeroKernel(gpytorch.kernels.Kernel):
    def __init__(self):
        super(ZeroKernel, self).__init__()

    def forward(self, x1, x2, diag=False, **params):
        assert x1.shape[0] == x2.shape[0]
        return torch.zeros(( x1.shape[0], x1.shape[1], x2.shape[1])).to(device)

# --- SE ---
class SEKernelLight(gpytorch.kernels.Kernel):

    def __init__(self, lengthscale=torch.tensor([1.0]), output_scale=torch.tensor(1.0)):
        super(SEKernelLight, self).__init__(batch_shape=(lengthscale.shape[0], ))
        self.length_scale = lengthscale.to(device)
        self.ard_num_dims = lengthscale.shape[-1]
        self.output_scale = torch.reshape(output_scale, (lengthscale.shape[0], output_scale.shape[-1])).to(device)
        self.postprocess_rbf = lambda dist_mat: self.output_scale * dist_mat.div_(-2).exp_()


    def forward(self, x1, x2, diag=False, **params):
        # returns LazyTensor of size num_particles*num_samples_x1*num_samples_x2
        if (
                x1.requires_grad
                or x2.requires_grad
                or (self.ard_num_dims is not None and self.ard_num_dims > 1)
                or diag
        ):
            x1_ = x1.div(self.length_scale)
            x2_ = x2.div(self.length_scale)
            return self.covar_dist(x1_, x2_, square_dist=True, diag=diag,
                                   dist_postprocess_func=self.postprocess_rbf,
                                   postprocess=True, **params)
        # TODO: self.output_scale *
        res= RBFCovariance().apply(x1, x2, self.length_scale,
                                     lambda x1, x2: self.covar_dist(x1, x2,
                                                                    diag=False,
                                                                    last_dim_is_batch=False,
                                                                    square_dist=True,
                                                                    dist_postprocess_func=self.postprocess_rbf,
                                                                    postprocess=False))
        return res


# --- Linear ---
class LinearKernelLight(gpytorch.kernels.Kernel):

    def __init__(self, variance):
        # variance is of shape num_particles*1*dim_particles
        super(LinearKernelLight, self).__init__(
            batch_shape=(variance.shape[0],),   # allow different variances per batch
        )
        self.variance = variance.to(device)
        self.link_k = gpytorch.kernels.LinearKernel(
            batch_shape=(variance.shape[0],),
            ard_num_dims=variance.shape[-1]
        )
        self.link_k.variance = torch.ones(
            self.link_k.variance.shape,
            device=device, requires_grad=False
        )


    def forward(self, x1, x2, diag=False, **params):
        '''
        x1 and x2 are from sizes
            num_particles * num_samples * dim_particles,
        where num_sampls for x1 and x2 can be (n_test, n_train+n_test)
        or (n_train, n_train).
        '''
        x1 = torch.mul(x1, self.variance)
        #res = self.link_k.forward(x1, x2, diag=diag, **params)
        return self.link_k.forward(x1, x2, diag=diag, **params)


# --- PeriodicKernelLight ---
class PeriodicKernelLight(gpytorch.kernels.Kernel):

    def __init__(self, ard_num_dims=None, batch_size= None,
                 periodic_length_scale=None,
                 periodic_output_scale=None,
                 period=None):
        self.ard_num_dims, self.batch_size = ard_num_dims, batch_size
        self.periodic_length_scale, self.period = periodic_length_scale, period
        self.periodic_output_scale = periodic_output_scale

        # infer sizes from inputs if not given
        if self.ard_num_dims is None:
            assert not (periodic_length_scale is None and period is None)
            if not self.periodic_length_scale is None:
                self.ard_num_dims = self.periodic_length_scale.shape[-1]
            elif not self.period is None:
                self.ard_num_dims = self.period.shape[-1]
        if self.batch_size is None:
            assert not (periodic_length_scale is None and period is None and periodic_output_scale is None)
            if not self.periodic_length_scale is None:
                self.batch_size = self.periodic_length_scale.shape[0]
            elif not self.period is None:
                self.batch_size = self.period.shape[0]
            elif not self.period is None:
                self.batch_size = self.periodic_output_scale.shape[0]

        # initialize hyper-params if not given
        if self.periodic_length_scale is None:
            self.periodic_length_scale = torch.tensor(np.ones((self.batch_size, 1, self.ard_num_dims))).float().to(device)
        else:
            self.periodic_length_scale = self.periodic_length_scale.view(self.batch_size, 1, self.ard_num_dims).to(device)

        if self.period is None:
            self.period = torch.tensor(np.ones((self.batch_size, 1, self.ard_num_dims))).float().to(device)
        else:
            self.period = self.period.view(self.batch_size, 1, self.ard_num_dims).to(device)

        if self.periodic_output_scale is None:
            self.periodic_output_scale = torch.tensor(np.ones((self.batch_size, 1, 1))).float().to(device)
        else:
            self.periodic_output_scale = self.periodic_output_scale.view(self.batch_size, 1, 1).to(device)

        super(PeriodicKernelLight, self).__init__(batch_shape=(self.batch_size, ))



    def forward(self, x1, x2, diag=False, **params):
        # apply length_scale
        x1_ = x1.div(self.period/math.pi)
        x2_ = x2.div(self.period/math.pi)
        # transpose to have batch * num_feat * num_samples
        x1_ = torch.transpose(x1_,-1,-2)
        x2_ = torch.transpose(x2_,-1,-2)
        periodic_length_scale = torch.transpose(self.periodic_length_scale.detach(),-1,-2)
        # expand last dimension
        x1_ = x1_.view((*x1_.shape, 1))
        x2_ = x2_.view((*x2_.shape, 1))
        periodic_length_scale = periodic_length_scale.view((*periodic_length_scale.shape, 1))
        # calculate element wise distance between pairwise inputs
        # diff is batch * num_feat * num_samples * num_samples
        diff = torch.cdist(x1_, x2_, p=1.0)
        # prevent divide by 0 errors
        diff.where(diff == 0, torch.as_tensor(1e-20))
        exp_dist = (torch.mean((diff.sin_()**2).div_(periodic_length_scale), dim=1)).exp()
        # NOTE: in the original formula, must be sum instead of mean
        return self.periodic_output_scale * exp_dist



# import positivity constraint
from gpytorch.constraints import Positive
class PeriodicKernel(gpytorch.kernels.Kernel):

    def __init__(self, ard_num_dims,
                 periodic_output_scale_prior=None, periodic_output_scale_constraint=Positive(),
                 periodic_length_scale_prior=None, periodic_length_scale_constraint=Positive(),
                 period_prior=None, period_constraint=Positive(),
                 **kwargs):
        super().__init__(**kwargs)

        self.ard_num_dims = ard_num_dims
        self.periodic_output_scale_constraint = periodic_output_scale_constraint
        self.periodic_length_scale_constraint = periodic_length_scale_constraint
        self.period_constraint = period_constraint

        # register the raw parameter
        self.register_parameter(name='periodic_output_scale_raw',
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))
        self.register_parameter(name='periodic_length_scale_raw',
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, self.ard_num_dims)))
        self.register_parameter(name='period_raw',
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, self.ard_num_dims)))


        # register the constraint
        self.register_constraint("periodic_length_scale_raw", periodic_length_scale_constraint)
        self.register_constraint("periodic_output_scale_raw", periodic_output_scale_constraint)
        self.register_constraint("period_raw", period_constraint)

        # register prior
        if periodic_output_scale_prior is not None:
            self.register_prior(
                "periodic_output_scale_prior",
                periodic_output_scale_prior,
                lambda m: m.length,
                lambda m, v : m._set_length(v),
            )
        if periodic_length_scale_prior is not None:
            self.register_prior(
                "periodic_length_scale_prior",
                periodic_length_scale_prior,
                lambda m: m.length,
                lambda m, v : m._set_length(v),
            )
        if period_prior is not None:
            self.register_prior(
                "period_prior",
                period_prior,
                lambda m: m.length,
                lambda m, v : m._set_length(v),
            )

    # now set up the 'actual' parameter
    @property
    def periodic_output_scale(self):
        # when accessing the parameter, apply the constraint transform
        return self.periodic_output_scale_constraint.transform(self.periodic_output_scale_raw)
    @property
    def periodic_length_scale(self):
        return self.periodic_length_scale_constraint.transform(self.periodic_length_scale_raw)
    @property
    def period(self):
        return self.period_constraint.transform(self.period_raw)

    @periodic_output_scale.setter
    def periodic_output_scale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.periodic_output_scale_raw)
        # when setting the parameter, transform the actual value to a raw one by applying the inverse transform
        self.initialize(periodic_output_scale_raw=self.periodic_output_scale_constraint.inverse_transform(value))
    @periodic_length_scale.setter
    def periodic_length_scale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.periodic_length_scale_raw)
        # when setting the parameter, transform the actual value to a raw one by applying the inverse transform
        self.initialize(periodic_length_scale_raw=self.periodic_length_scale_constraint.inverse_transform(value))
    @period.setter
    def period(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.period_raw)
        # when setting the parameter, transform the actual value to a raw one by applying the inverse transform
        self.initialize(period_raw=self.period_constraint.inverse_transform(value))

    # this is the kernel function
    def forward(self, x1, x2, **params):
        # apply period
        x1_ = x1.div(self.period) * math.pi
        x2_ = x2.div(self.period) * math.pi
        # calculate the distance between inputs
        diff = x1_ - x2_
        # prevent divide by 0 errors
        diff.where(diff == 0, torch.as_tensor(1e-20))
        exp_dist = (torch.exp(-0.5 * torch.sum(torch.sin(diff)**2).div(self.periodic_length_scale)))
        return self.periodic_output_scale * exp_dist


# --- ---
class HomoskedasticNoiseLight(gpytorch.likelihoods.noise_models._HomoskedasticNoiseBase):

    def __init__(self, noise_var, *params, **kwargs):
        self.noise_var = noise_var.to(device)
        self._modules = {}
        self._parameters = {}

    @property
    def noise(self):
        return self.noise_var

    @noise.setter
    def noise(self, value):
        self.noise_var = value.to(device) # double check

class FixedGaussianNoiseLight(gpytorch.likelihoods.noise_models.FixedGaussianNoise):

    def __init__(self, noise_var):
        super().__init__(noise=noise_var.to(device))
        self._modules = {}
        self._parameters = {}


class GaussianLikelihoodLight(gpytorch.likelihoods._GaussianLikelihoodBase):
    def __init__(self, noise_var, batch_shape=torch.Size()):
        self.batch_shape = batch_shape
        self._modules = {}
        self._parameters = {}

        noise_covar = HomoskedasticNoiseLight(noise_var.to(device)) # double check
        super().__init__(noise_covar=noise_covar)

    @property
    def noise(self):
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value):
        self.noise_covar.noise = value

    def expected_log_prob(self, target, input, *params, **kwargs):
        mean, variance = input.mean, input.variance
        noise = self.noise_covar.noise
        res = ((target - mean) ** 2 + variance) / noise + noise.log() + math.log(2 * math.pi)
        res = res.mul(-0.5).sum(-1)
        return res

    def forward(self, *params, shape=None, noise=None, **kwargs):
        if noise is not None:
            return super().forward(*params, shape=shape, noise=noise.to(device), **kwargs)
        else:
            return super().forward(*params, shape=shape, noise=noise.to(device), **kwargs)


class FixedNoiseGaussianLikelihoodLight(gpytorch.likelihoods._GaussianLikelihoodBase):
    def __init__(self, noise_var, second_noise_var=None,
                learn_additional_noise=False, batch_shape=torch.Size()):
        assert ((second_noise_var is None) ^ learn_additional_noise)
        self.batch_shape = batch_shape
        self._modules = {}
        self._parameters = {}

        noise_covar = FixedGaussianNoiseLight(noise_var.to(device))
        super().__init__(noise_covar=noise_covar)

        if learn_additional_noise:
            self.second_noise_covar = HomoskedasticNoiseLight(
                batch_shape=batch_shape
            )
        else:
            self.second_noise_covar = None


    def expected_log_prob(self, target, input, *params, **kwargs):
        mean, variance = input.mean, input.variance
        noise = self.noise
        res = ((target - mean) ** 2 + variance) / noise + noise.log() + math.log(2 * math.pi)
        res = res.mul(-0.5).sum(-1)
        return res

    @property
    def noise(self):
        return self.noise_covar.noise + self.second_noise

    @noise.setter
    def noise(self, value):
        self.noise_covar.initialize(noise=value)

    @property
    def second_noise(self):
        if self.second_noise_covar is None:
            return 0
        else:
            return self.second_noise_covar.noise

    @second_noise.setter
    def second_noise(self, value):
        if self.second_noise_covar is None:
            raise RuntimeError(
                "Attempting to set secondary learned noise for FixedNoiseGaussianLikelihood, "
                "but learn_additional_noise must have been False!"
            )
        self.second_noise_covar.initialize(noise=value)





class LearnedGPRegressionModel(gpytorch.models.ExactGP):
    """GP model which can take a learned mean and learned kernel function."""
    def __init__(self, train_x, train_y, likelihood, covar_module,
                mean_module=None, learned_kernel=None, learned_mean=None):
        super(LearnedGPRegressionModel, self).__init__(train_x, train_y, likelihood)
        if mean_module is None:
            self.mean_module = gpytorch.means.ZeroMean()
        else:
            self.mean_module = mean_module
        self.covar_module = covar_module
        self.learned_kernel = learned_kernel
        self.learned_mean = learned_mean
        self.likelihood = likelihood


    def forward(self, x):
        # feed through kernel NN
        if self.learned_kernel is not None:
            projected_x = self.learned_kernel(x)
        else:
            projected_x = x
        covar_x = self.covar_module(projected_x)
        # feed through mean module
        if self.learned_mean is not None: # mean is NN
            mean_x = self.learned_mean(x).squeeze()
        else:
            mean_x = self.mean_module(x).squeeze()
        res = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return res

    def prior(self, x):
        self.train()
        return self.__call__(x)

    def posterior(self, x):
        self.train()    # to clear the cache
        self.eval()
        return self.__call__(x)

    def kl(self, x):
        return torch.distributions.kl.kl_divergence(self.posterior(x), self.prior(x))

    def pred_dist(self, x):
        self.train()    # to clear the cache
        self.eval()
        return self.likelihood(self.__call__(x))

    def pred_ll(self, x, y):
        pred_dist = self.pred_dist(x)
        return pred_dist.log_prob(y)


    def __call__(self, *args, **kwargs):
        with gpytorch.settings.fast_pred_var(False):
            return super().__call__(*args, **kwargs)


# --------------
def exact_pred(prediction_strategy, joint_mean, joint_covar):
    # Find the components of the distribution that contain test data
    test_mean = joint_mean[..., prediction_strategy.num_train :]
    # For efficiency - we can make things more efficient
    if joint_covar.size(-1) <= gpytorch.settings.max_eager_kernel_size.value():
        test_covar = joint_covar[..., prediction_strategy.num_train :, :].to_dense()
        test_test_covar = test_covar[..., prediction_strategy.num_train :]
        test_train_covar = test_covar[..., : prediction_strategy.num_train]
    else:
        test_test_covar = joint_covar[..., prediction_strategy.num_train :, prediction_strategy.num_train :]
        test_train_covar = joint_covar[..., prediction_strategy.num_train :, : prediction_strategy.num_train]
    if torch.isnan(test_mean).any():
        print('Nan in exact pred test_mean')
        raise NotImplementedError
    # if torch.isnan(test_test_covar).any():
    #     print('Nan in exact pred test_test_covar')
    #     raise NotImplementedError
    # if torch.isnan(test_train_covar).any():
    #     print('Nan in  exact pred test_train_covar')
    #     raise NotImplementedError
    return (
        prediction_strategy.exact_predictive_mean(test_mean, test_train_covar),
        prediction_strategy.exact_predictive_covar(test_test_covar, test_train_covar),
    )

# --------------

from gpytorch.models.approximate_gp import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


class LearnedGPRegressionModelApproximate(ApproximateGP):
    """GP model which can take a learned mean and learned kernel function."""
    def __init__(self, train_x, train_y, likelihood, learned_kernel=None, learned_mean=None, mean_module=None,
                 covar_module=None):

        self.n_train_samples = train_x.shape[0]

        variational_distribution = CholeskyVariationalDistribution(self.n_train_samples)
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution,
                                                   learn_inducing_locations=False)
        super().__init__(variational_strategy)

        if mean_module is None:
            self.mean_module = gpytorch.means.ZeroMean()
        else:
            self.mean_module = mean_module

        self.covar_module = covar_module

        self.learned_kernel = learned_kernel
        self.learned_mean = learned_mean
        self.likelihood = likelihood

    def forward(self, x):
        # feed through kernel NN
        if self.learned_kernel is not None:
            projected_x = self.learned_kernel(x)
        else:
            projected_x = x

        # feed through mean module
        if self.learned_mean is not None:
            mean_x = self.learned_mean(x).squeeze()
        else:
            mean_x = self.mean_module(projected_x).squeeze()

        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def prior(self, x):
        return self.forward(x)

    def kl(self):
        return self.variational_strategy.kl_divergence()

    def pred_dist(self, x):
        self.train()    # to clear the cache
        self.eval()
        return self.likelihood(self.__call__(x))

    def pred_ll(self, x, y):
        variational_dist_f = self.__call__(x)
        return self.likelihood.expected_log_prob(y, variational_dist_f).sum(-1)

    @property
    def variational_distribution(self):
        return self.variational_strategy._variational_distribution
