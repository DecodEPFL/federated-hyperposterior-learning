import copy, gpytorch, torch, math, warnings
import numpy as np
import torch.nn.functional as F
from pyro.distributions import Normal, Independent
from collections import OrderedDict

from server.util import warning_show, warning_format, print_gpvec_prior_params, WrapLogger, softplus_inverse
from server.models import LearnedGPRegressionModel, ConstantMeanLight, SEKernelLight, GaussianLikelihoodLight, \
    FixedNoiseGaussianLikelihoodLight, VectorizedModel, CatDist, NeuralNetworkVectorized, PeriodicKernelLight, LinearKernelLight
from config import device

warnings.formatwarning = warning_format
#warnings.showwarning = warning_show

def _filter(dict, str):
    result = OrderedDict()
    for key, val in dict.items():
        if str in key:
            result[key] = val
    return result


class VectorizedGP(VectorizedModel):

    def __init__(
        self, input_dim, likelihood_str, covar_module_str, mean_module_str,
        mean_nn_layers=(32, 32), kernel_nn_layers=(32, 32), feature_dim=None,
        nonlinearity_hidden_m=torch.tanh, nonlinearity_hidden_k=torch.tanh,
        nonlinearity_output_m=None, nonlinearity_output_k=None,
        optimize_noise=True, noise_std=None,
        optimize_lengthscale=True, lengthscale_fix=None,
        logger=None):
        # feature_dim: (int) output dimensionality of NN feature map for kernel function
        # if optimize_noise is False, the noise std must be given by noise_std. (similarly for lengthscale)
        # NOTE: softplus not applied on noise_std and on lengthscale_fix => true values are passed
        super().__init__(input_dim, 1)
        assert likelihood_str in ['Gaussian', 'FixedNoise']
        self.mean_module_str, self.covar_module_str = mean_module_str, covar_module_str
        self.mean_nn_layers, self.kernel_nn_layers = mean_nn_layers, kernel_nn_layers
        self.optimize_noise, self.optimize_lengthscale = optimize_noise, optimize_lengthscale
        self.likelihood_str = likelihood_str
        self.nonlinearity_hidden_m, self.nonlinearity_output_m=nonlinearity_hidden_m, nonlinearity_output_m
        self.nonlinearity_hidden_k, self.nonlinearity_output_k=nonlinearity_hidden_k, nonlinearity_output_k
        self.logger=WrapLogger(logger) if not isinstance(logger, WrapLogger) else logger

        if self.optimize_lengthscale:
            assert lengthscale_fix is None, "fix lengthscale or optimize it, cannot do both."
        else:
            assert not (lengthscale_fix is None), "must provide lengthscale if not optimizing it."

        # linear mean as NN without hidden layers
        requires_bias = {'out':True, 'hidden':True}
        if self.mean_module_str.startswith('linear'):
            self.nonlinearity_hidden_m, self.nonlinearity_output_m=None, None
            self.mean_nn_layers =[]
            if self.mean_module_str == 'linear_no_bias':
                requires_bias = {'out':False, 'hidden':False}
            self.mean_module_str = 'NN'

        # feature dim for NN kernel
        if (self.covar_module_str in ['NN', 'SNN']) and (feature_dim is None):
            self.logger.info('[WARNING] should provide output degree of NN. using default = 1')
        self.feature_dim = 1 if feature_dim is None else feature_dim
        self._params = OrderedDict()


        # --- define mean ---
        if self.mean_module_str == 'NN':
            self.mean_nn = self._param_module(
                'mean_nn',
                NeuralNetworkVectorized(
                    input_dim, 1, layer_sizes=self.mean_nn_layers,
                    nonlinearity_hidden=self.nonlinearity_hidden_m,
                    nonlinearity_output=self.nonlinearity_output_m,
                    requires_bias=requires_bias)
            )
        elif self.mean_module_str == 'constant':
            self.constant_mean = self._param(
                'constant_mean', torch.zeros(1, 1))
        elif self.mean_module_str == 'zero':
            self.constant_mean = torch.tensor([[0]]).float().to(device)
            self.constant_mean.requires_grad=False
        else:
            raise NotImplementedError


        # --- define kernel ---
        if self.covar_module_str in ['NN', 'SNN']:
            self.kernel_nn = self._param_module(
                'kernel_nn',
                NeuralNetworkVectorized(
                    input_dim, self.feature_dim,
                    layer_sizes=self.kernel_nn_layers,
                    nonlinearity_hidden=self.nonlinearity_hidden_k,
                    nonlinearity_output=self.nonlinearity_output_k,
                    requires_bias={'out':False, 'hidden':True}) # NOTE: do not need bias on kernel NN output
            )
            if self.optimize_lengthscale:
                self.lengthscale_raw = self._param(
                    'lengthscale_raw', torch.zeros(1, self.feature_dim))
            else:
                self.lengthscale_raw = torch.tensor(
                    softplus_inverse(lengthscale_fix)).float().to(device)
                self.lengthscale_raw.requires_grad=False

            if self.covar_module_str == 'SNN':
                self.outputscale_raw = self._param(
                    'outputscale_raw', torch.zeros(1, 1))
            else:
                self.outputscale_raw = torch.tensor(
                    [[softplus_inverse(1)]]).float().to(device)
                self.outputscale_raw.requires_grad=False

        elif self.covar_module_str == 'SE':
            if self.optimize_lengthscale:
                self.lengthscale_raw = self._param(
                    'lengthscale_raw', torch.zeros(1, input_dim))
            else:
                self.lengthscale_raw = torch.tensor(
                    softplus_inverse(lengthscale_fix)).float().to(device)
                self.lengthscale_raw.requires_grad=False
            self.outputscale_raw = self._param(
                'outputscale_raw', torch.zeros(1, 1))

        elif self.covar_module_str == 'linear':
            self.variance_raw = self._param('variance_raw', torch.zeros(1, input_dim))

        elif self.covar_module_str == 'periodic':
            self.periodic_length_scale_raw = self._param(
                'periodic_length_scale_raw', torch.zeros(1, input_dim))
            self.periodic_output_scale_raw = self._param(
                'periodic_output_scale_raw', torch.zeros(1, 1))
            self.period_raw = self._param(
                'period_raw', torch.zeros(1, input_dim))

        elif not self.covar_module_str == 'zero':
            raise NotImplementedError

        if self.optimize_noise:
            self.noise_raw = self._param(
                'noise_raw', torch.zeros(1, 1))
        elif noise_std is not None:
            self.noise_raw = torch.tensor(
                [[softplus_inverse(noise_std)]]).float().to(device).reshape((1,1))
            self.noise_raw.requires_grad=False
        else:
            raise NotImplementedError


    def forward(self, x_train, y_train, train=True, prior=False):
        assert x_train.ndim == 3

        # -- mean --
        if self.mean_module_str == 'NN':
            learned_mean = self.mean_nn
            mean_module = None
        elif self.mean_module_str in ['constant', 'zero']:
            learned_mean = None
            mean_module = ConstantMeanLight(self.constant_mean)
        elif self.mean_module_str == 'linear_no_bias':
            learned_mean = None
            mean_module = gpytorch.means.LinearMean(input_size=self.weights.shape[1], bias=False)
            # mean_module._parameters['weights'] = self.weights.clone().detach().requires_grad_(True)
            mean_module.weights = torch.nn.Parameter(self.weights.clone().detach().requires_grad_(True))
        elif self.mean_module_str == 'linear':
            learned_mean = None
            mean_module = gpytorch.means.LinearMean(input_size=self.weights.shape[1], bias=True)
            #mean_module._parameters['weights'] = self.weights.clone().detach().requires_grad_(True)
            #mean_module._parameters['bias'] = self.bias.clone().detach().requires_grad_(True)
            mean_module.weights = torch.nn.Parameter(self.weights.clone().detach().requires_grad_(True))
            mean_module.bias = torch.nn.Parameter(self.bias.clone().detach().requires_grad_(True))
        else:
            raise NotImplementedError

        # -- covar --
        if (self.covar_module_str=='NN') or (self.covar_module_str=='SNN'):
            learned_kernel = self.kernel_nn
            lengthscale = F.softplus(self.lengthscale_raw)
            outputscale = F.softplus(self.outputscale_raw)
            lengthscale = lengthscale.view(lengthscale.shape[0], 1, self.feature_dim)
            outputscale = outputscale.view(outputscale.shape[0], 1, 1)
            if torch.isnan(lengthscale).any():
                self.logger.info('err nan lengthscale ', self.lengthscale_raw)
                raise NotImplementedError
            if torch.isnan(outputscale).any():
                self.logger.info('err nan outputscale ', self.outputscale_raw)
                raise NotImplementedError

            for i in np.arange(learned_kernel.n_layers):
                linvec = getattr(learned_kernel, 'fc_%i'%(i+1))
                if torch.isnan(linvec.weight).any():
                    self.logger.info('err nan in kernel nn weights hidden')
                    raise NotImplementedError
                if torch.isnan(linvec.bias).any():
                    self.logger.info('err nan in kernel nn bias hidden')
                    raise NotImplementedError
            linvec =getattr(learned_kernel, 'out')
            if torch.isnan(linvec.weight).any():
                self.logger.info('err nan in kernel nn weights out')
                raise NotImplementedError
            if torch.isnan(linvec.bias).any():
                self.logger.info('err nan in kernel nn bias out')
                raise NotImplementedError

            covar_module = SEKernelLight(lengthscale, output_scale=outputscale)
            #covar_module = SEKernelLight(lengthscale)
        else:
            learned_kernel = None
            if self.covar_module_str == 'linear':
                variance = F.softplus(self.variance_raw).to(device)
                variance = variance.view(variance.shape[0], 1, self.input_dim)
                covar_module = LinearKernelLight(variance)
                # print('var raw ', self.variance_raw.shape)
            elif self.covar_module_str == 'SE':
                lengthscale = F.softplus(self.lengthscale_raw)
                outputscale = F.softplus(self.outputscale_raw)
                #if len(lengthscale.shape) > 1:
                #    lengthscale = lengthscale.view(lengthscale.shape[0], 1, lengthscale.shape[1])
                #else:
                #    lengthscale = lengthscale.view(lengthscale.shape[0], 1, self.input_dim)
                lengthscale = lengthscale.view(lengthscale.shape[0], 1, self.input_dim)
                outputscale = outputscale.view(outputscale.shape[0], 1, 1)
                covar_module = SEKernelLight(lengthscale, output_scale=outputscale)
            elif self.covar_module_str == 'periodic':
                periodic_length_scale = F.softplus(self.periodic_length_scale_raw)
                periodic_output_scale = F.softplus(self.periodic_output_scale_raw)
                period = F.softplus(self.period_raw)
                periodic_length_scale = periodic_length_scale.view(
                    periodic_length_scale.shape[0], 1, self.input_dim)
                period = period.view(period.shape[0], 1, self.input_dim)
                covar_module = PeriodicKernelLight(
                    periodic_length_scale=periodic_length_scale,
                    periodic_output_scale=periodic_output_scale,
                    period=period)
            elif self.covar_module_str == 'zero':
                #covar_module = ZeroKernel()
                covar_module = gpytorch.kernels.LinearKernel()
                covar_module.raw_variance = torch.nn.Parameter(
                    torch.tensor([[float('-inf')]]).requires_grad_(False).to(device))
                #covar_module.variance = torch.tensor([[0]]).float().requires_grad_(False).to(device) # set to 0, fixed, no grad
            else:
                raise NotImplementedError("unknown covariance module " + self.covar_module_str)
        # -- noise on likelihood --
        noise = F.softplus(self.noise_raw)

        if self.likelihood_str == 'Gaussian':
            likelihood = GaussianLikelihoodLight(noise) # fixed noise on all test points
        elif self.likelihood_str == 'FixedNoise':
            likelihood = FixedNoiseGaussianLikelihoodLight(
                noise, second_noise_var=None,
                learn_additional_noise=False) # TODO: enable learning additional noise

        # -- form the gp --
        gp = LearnedGPRegressionModel(
            x_train, y_train, likelihood, mean_module=mean_module, covar_module=covar_module,
            learned_mean=learned_mean, learned_kernel=learned_kernel)
        if prior:
            gp.train()
            likelihood.train()
            return gp, likelihood
        else:
            if train:
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
                output = gp(x_train) # MultivariateNormal(m(x_train), K(x_train, x_train)) => PRIOR at x_train
                # NOTE: output is not affected by noise in the likelihood and has an ill-conditioned cov if train samples are corr
                # avoid negative or very large entries in the cov matrix
                output.covariance_matrix = torch.clamp(output.covariance_matrix,
                                                       min=0, max=1e4)
                if torch.isnan(output.covariance_matrix).any():
                    raise NotImplementedError("unknown reason for nan in gp output cov in random_gp forward")
                if torch.isnan(output.mean).any():
                    raise NotImplementedError("unknown reason for nan in gp output mean in random_gp forward")
                # check condition number
                cond_num = torch.linalg.cond(likelihood(output).covariance_matrix, p=2)

                if (cond_num >=1e6).any():
                    self.logger.info('[WARN] ill-conditioned covariance matrix between train inputs after adding noise (condition number = ', cond_num, '). increasing jitter')
                with gpytorch.settings.cholesky_jitter(1e-1): # increase jitter to avoid non-psd cov
                    mll = mll(output, y_train)
                if torch.isnan(mll).any():
                    # check condition number
                    cond_num = torch.linalg.cond(likelihood(output).covariance_matrix, p=2)
                    if cond_num >=1e3:
                        raise IllCondException
                    else:
                        self.logger.info(print_gpvec_prior_params(self))
                        raise NotImplementedError("unknown error in random_gp forward (cond num = {:6.2f}".format(cond_num))
                return likelihood(output), mll
            else:  # --> eval
                #gp._clear_cache()          # to clear the cache
                #likelihood._clear_cache()  # to clear the cache
                gp.training = False
                likelihood.training = False
                return gp, likelihood

    def parameter_shapes(self):
        return OrderedDict([(name, param.shape) for name, param in self.named_parameters().items()])

    def named_parameters(self):
        return self._params

    def _param_module(self, name, module):
        assert type(name) == str
        assert hasattr(module, 'named_parameters')

        for param_name, param in module.named_parameters().items():
            self._param(name + '.' + param_name, param)
        return module

    def _param(self, name, tensor):
        assert type(name) == str
        assert isinstance(tensor, torch.Tensor)
        assert name not in list(self._params.keys())
        if not device.type == tensor.device.type:
            tensor = tensor.to(device)
        self._params[name] = tensor
        return tensor

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _RandomGPBase:

    def __init__(self, input_dim, feature_dim,
                 prior_factor,
                 optimize_noise, noise_std, likelihood_str,
                 covar_module_str, mean_module_str,
                 mean_nn_layers=(32, 32), kernel_nn_layers=(32, 32),
                 hyper_prior_dict={},
                 nonlinearity_hidden_m=torch.tanh, nonlinearity_hidden_k=torch.tanh,
                 nonlinearity_output_m=None, nonlinearity_output_k=torch.tanh,
                 output_dim=1, logger=None, **kwargs):
        self.logger=WrapLogger(logger) if not isinstance(logger, WrapLogger) else logger
        # hyper-prior is a dictionary of parameter name and std
        self._params = OrderedDict()
        self._param_dists = OrderedDict()

        self.prior_factor = prior_factor
        self.gp = VectorizedGP(
            input_dim, feature_dim=feature_dim,
            covar_module_str=covar_module_str, mean_module_str=mean_module_str,
            mean_nn_layers=mean_nn_layers, kernel_nn_layers=kernel_nn_layers,
            nonlinearity_hidden_m=nonlinearity_hidden_m,
            nonlinearity_hidden_k=nonlinearity_hidden_k,
            nonlinearity_output_m=nonlinearity_output_m,
            nonlinearity_output_k=nonlinearity_output_k,
            optimize_noise=optimize_noise, noise_std=noise_std,
            likelihood_str=likelihood_str, logger=self.logger, **kwargs)


        for name, shape in self.gp.parameter_shapes().items(): #TODO: equal hyper-prior on lengthscale and period of all input dimensions
            # -- constant_mean --
            if 'constant_mean' in name: # default: 0, 1
                mean_p_loc   = hyper_prior_dict.get(
                    'constant_mean_loc')*torch.ones(1).to(device)
                mean_p_scale = hyper_prior_dict.get(
                    'constant_mean_scale')*torch.ones(1).to(device)
                self._param_dist(name, Normal(mean_p_loc, mean_p_scale).to_event(1))

            # -- lengthscale_raw --
            elif 'lengthscale_raw' in name: # default: 1,3
                lengthscale_raw_loc =   hyper_prior_dict.get(
                    'lengthscale_raw_loc')*torch.ones(shape[-1]).to(device)
                lengthscale_raw_scale = hyper_prior_dict.get(
                    'lengthscale_raw_scale')*torch.ones(shape[-1]).to(device)
                self._param_dist(name, Normal(lengthscale_raw_loc, lengthscale_raw_scale).to_event(1))

            # -- outputscale_raw --
            elif 'outputscale_raw' in name: # default: 1, 3
                outputscale_raw_loc =   hyper_prior_dict.get(
                    'outputscale_raw_loc')*torch.ones(shape[-1]).to(device)
                outputscale_raw_scale = hyper_prior_dict.get(
                    'outputscale_raw_scale')*torch.ones(shape[-1]).to(device)
                self._param_dist(name, Normal(outputscale_raw_loc, outputscale_raw_scale).to_event(1))

            # -- variance_raw --
            elif 'variance_raw' in name: # default: 5, 1
                variance_raw_loc   = hyper_prior_dict.get(
                    'variance_raw_loc')*torch.ones(shape[-1]).to(device)
                variance_raw_scale = hyper_prior_dict.get(
                    'variance_raw_scale')*torch.ones(shape[-1]).to(device)
                self._param_dist(name, Normal(variance_raw_loc, variance_raw_scale).to_event(1))

            # -- periodic_length_scale_raw --
            elif 'periodic_length_scale_raw' in name: # default: 5, 1
                periodic_length_scale_raw_loc =   hyper_prior_dict.get(
                    'periodic_length_scale_raw_loc')*torch.ones(shape[-1]).to(device)
                periodic_length_scale_raw_scale = hyper_prior_dict.get(
                    'periodic_length_scale_raw_scale')*torch.ones(shape[-1]).to(device)
                self._param_dist(name, Normal(periodic_length_scale_raw_loc, periodic_length_scale_raw_scale).to_event(1))

            # -- periodic_output_scale_raw --
            elif 'periodic_output_scale_raw' in name: # default: 5, 1
                periodic_output_scale_raw_loc =   hyper_prior_dict.get(
                    'periodic_output_scale_raw_loc')*torch.ones(shape[-1]).to(device)
                periodic_output_scale_raw_scale = hyper_prior_dict.get(
                    'periodic_output_scale_raw_scale')*torch.ones(shape[-1]).to(device)
                self._param_dist(name, Normal(periodic_output_scale_raw_loc, periodic_output_scale_raw_scale).to_event(1))

            # -- period_raw --
            elif 'period_raw' in name: # default: 5, 1
                period_raw_loc =   hyper_prior_dict.get(
                    'period_raw_loc')*torch.ones(shape[-1]).to(device)
                period_raw_scale = hyper_prior_dict.get(
                    'period_raw_scale')*torch.ones(shape[-1]).to(device)
                self._param_dist(name, Normal(period_raw_loc, period_raw_scale).to_event(1))


            # -- noise_raw --
            elif 'noise_raw' in name and optimize_noise:
                noise_raw_loc   = hyper_prior_dict.get(
                    'noise_raw_loc')*torch.ones(1).to(device)
                noise_raw_scale = hyper_prior_dict.get(
                    'noise_raw_scale')*torch.ones(1).to(device)
                self._param_dist(name, Normal(noise_raw_loc, noise_raw_scale).to_event(1))

            # -- NN --
            elif 'mean_nn' in name or 'kernel_nn' in name: # TODO: edit here
                weight_default_dist = False
                bias_default_dist = False
                name_split = name.split('.')
                layer_lin_vec = getattr(getattr(self.gp, name_split[0]),
                                        name_split[1])
                if name_split[2]=='weight':
                    dist = layer_lin_vec.weight_dist
                    if dist is None: # Kaiming was not used. set according to the hyper-prior
                        weight_default_dist = True
                elif name_split[2]=='bias':
                    dist = layer_lin_vec.bias_dist
                    # reduce regularization on bias
                    dist.scale = dist.scale*20
                    #dist specified with dist.mean, dist.scale, dist.covariance)
                    if dist is None:
                        bias_default_dist = True
                else:
                    self.logger.info('[ERROR] name ' + name + ' not recognized.')

                # set dist according to the hyper-prior
                if weight_default_dist:
                    if not ('weights_loc' in hyper_prior_dict.keys() and 'weights_scale' in hyper_prior_dict.keys()):
                        self.logger.info('[WARNING]: hyper prior for weights was not provided. Replaced by default.')
                    if ('mean_nn' in name and mean_nn_layers==[]) or ('kernel_nn' in name and kernel_nn_layers==[]):
                        dim = input_dim*output_dim
                    else:
                        dim = shape
                    mean = hyper_prior_dict.get('weights_loc', 0)*torch.ones(dim).to(device)
                    std = hyper_prior_dict.get('weights_scale', 1)*torch.ones(dim).to(device)
                    #mean = torch.zeros(shape).to(device)
                    #std = hyper_prior_dict.get('weights', 1) * torch.ones(shape).to(device)
                    dist = Normal(mean, std)
                if bias_default_dist:
                    if not ('bias_loc' in hyper_prior_dict.keys() and 'bias_scale' in hyper_prior_dict.keys()):
                        self.logger.info('[WARNING]: hyper prior for bias was not provided. Replaced by default.')
                        if 'out' in name:
                            dim = output_dim
                        elif 'fc' in name:
                            dim = shape
                        else:
                            raise NotImplementedError
                    mean   = hyper_prior_dict.get('bias_loc',   0)*torch.ones(dim).to(device)
                    std = hyper_prior_dict.get('bias_scale', 1)*torch.ones(dim).to(device)
                    #mean = torch.zeros(shape).to(device)
                    #std = hyper_prior_dict.get('bias', 1) * torch.ones(shape).to(device)
                    dist = Normal(mean, std)
                self._param_dist(name, dist.to_event(1))
            else:
                self.logger.info('[ERROR] name ' + name + ' not recognized.')

        # check that parameters in prior and gp modules are aligned
        for param_name_gp, param_name_prior in zip(self.gp.named_parameters().keys(), self._param_dists.keys()):
            assert param_name_gp == param_name_prior

        self.hyper_prior = CatDist(self._param_dists.values())

    def sample_params_from_prior(self, shape=torch.Size()):
        return self.hyper_prior.sample(shape)

    def sample_fn_from_prior(self, shape=torch.Size()):
        params = self.sample_params_from_prior(shape=shape)
        fn_vecgp = self.get_forward_fn(params)
        return fn_vecgp

    def get_forward_fn(self, params):
        gp_model = copy.deepcopy(self.gp)
        gp_model.set_parameters_as_vector(params)
        return gp_model

    def _param_dist(self, name, dist):
        assert type(name) == str
        assert isinstance(dist, torch.distributions.Distribution)
        dist.base_dist.loc = dist.base_dist.loc.to(device)
        dist.base_dist.scale = dist.base_dist.scale.to(device)
        if name in list(self._param_dists.keys()):
            self.logger.info('[WARNING] name ' + name + 'was already in param dists')
        #assert name not in list(self._param_dists.keys()) #TODO
        assert hasattr(dist, 'rsample')
        self._param_dists[name] = dist

        return dist

    def _log_prob_hyper_prior(self, params):
        return self.hyper_prior.log_prob(params)

    def _log_prob_likelihood(self, *args):
        raise NotImplementedError

    def log_prob(self, *args):
        raise NotImplementedError

    def parameter_shapes(self):
        param_shapes_dict = OrderedDict()
        for name, dist in self._param_dists.items():
            param_shapes_dict[name] = dist.event_shape
        return param_shapes_dict

class RandomGP(_RandomGPBase):

    def _log_prob_likelihood(self, params, x_data, y_data):
        fn_vecgp = self.get_forward_fn(params)
        _, mll = fn_vecgp(x_data, y_data)
        return mll

    def log_prob(self, params, x_data, y_data):
        return self.prior_factor * self._log_prob_hyper_prior(params) + self._log_prob_likelihood(params, x_data, y_data)


class HyperPosterior(_RandomGPBase):

    def _log_prob_likelihood(self, params, train_data_tuples):
        fn_vecgp = self.get_forward_fn(params) # fn_vecgp is a VectorizedGP with params sampled from the current hyper-post approx
        num_datasets = len(train_data_tuples)
        dataset_sizes = torch.tensor([train_x.shape[-2] for train_x, _ in train_data_tuples]).float().to(device)
        harmonic_mean_dataset_size = 1. / (torch.mean(1. / dataset_sizes))
        pre_factor = harmonic_mean_dataset_size / (harmonic_mean_dataset_size + num_datasets)

        mlls = []
        for x_train, y_train in train_data_tuples:
            _, mll = fn_vecgp(x_train, y_train)
            mlls.append(mll)
        mlls = torch.stack(mlls, dim=-1)
        return pre_factor * torch.sum(mlls, dim=-1)

    def log_prob(self, params, train_data_tuples):
        return self.prior_factor * self._log_prob_hyper_prior(params) + self._log_prob_likelihood(params, train_data_tuples)

    def comb_weights(self, params, x_train, y_train, weights_hyp_prior=None):
        num_params = params.shape[0]
        if weights_hyp_prior is None:
            weights_hyp_prior = torch.ones(num_params).to(device)
        x_train = x_train.view(torch.Size((1,)) + x_train.shape).repeat(num_params, 1, 1)
        y_train = y_train.view(torch.Size((1,)) + y_train.shape).repeat(num_params, 1)
        # x_train size is torch.Size([num_particles, num_samples, num_features])
        # y_train size is torch.Size([num_particles, num_samples])

        fn_vecgp = self.get_forward_fn(params)
        _, mll = fn_vecgp(x_train, y_train)     # mll size is torch.Size([num_particles])
        mll = mll - torch.max(mll)
        comb_weights = torch.mul(
            torch.exp(x_train.shape[1] * mll),
            weights_hyp_prior
        )
        comb_weights = comb_weights/torch.sum(comb_weights)
        return comb_weights

class RandomGPPosterior(torch.nn.Module):
    '''
    Gaussian VI posterior on the GP-Prior parameters
    '''

    def __init__(self, named_param_shapes, init_std=0.1, cov_type='full', param_dists=None):
        # if param_dists is not None, initialize VI with the hyper-prior
        super().__init__()

        assert cov_type in ['diag', 'full']

        self.param_idx_ranges = OrderedDict()

        idx_start = 0
        for name, shape in named_param_shapes.items():
            assert len(shape) == 1
            idx_end = idx_start + shape[0]
            self.param_idx_ranges[name] = (idx_start, idx_end)
            idx_start = idx_end

        param_shape = torch.Size((idx_start,))

        # initialize VI
        if param_dists is None: # initialize randomly
            self.loc = torch.nn.Parameter(
                torch.normal(0.0, init_std, size=param_shape, device=device))
            if cov_type == 'diag':
                self.scale = torch.nn.Parameter(
                    torch.normal(math.log(0.1), init_std, size=param_shape, device=device))
                self.dist_fn = lambda: Normal(self.loc, self.scale.exp()).to_event(1)
            if cov_type == 'full':
                self.tril_cov = torch.nn.Parameter(
                    torch.diag(torch.ones(param_shape, device=device).uniform_(0.05, 0.1)))
                self.dist_fn = lambda: torch.distributions.MultivariateNormal(
                    loc=self.loc, scale_tril=torch.tril(self.tril_cov))
        else: # initialize to hyper-prior
            # go through all parameters by name and initialize
            loc=[]
            scale = []
            for name, shape in named_param_shapes.items():
                loc.append(param_dists[name].mean)
                scale.append(math.sqrt(param_dists[name].variance))
            self.loc = torch.nn.Parameter(torch.tensor(loc).float().to(device))
            if cov_type == 'diag':
                self.scale = torch.nn.Parameter(torch.tensor(scale).float().to(device))
                self.dist_fn = lambda: Normal(self.loc, self.scale).to_event(1)
            if cov_type == 'full':
                raise NotImplementedError


    def forward(self):
        return self.dist_fn()

    def rsample(self, sample_shape=torch.Size()):
        return self.forward().rsample(sample_shape)

    def sample(self, sample_shape=torch.Size()):
        return self.forward().sample(sample_shape)

    def log_prob(self, value):
        return self.forward().log_prob(value)

    @property
    def mode(self):
        return self.mean

    @property
    def mean(self):
        return self.forward().mean

    @property
    def stddev(self):
        return self.forward().stddev

    def entropy(self):
        return self.forward().entropy()

    @property
    def mean_stddev_dict(self):
        mean = self.mean
        stddev = self.stddev
        with torch.no_grad():
            return OrderedDict(
                [(name, (mean[idx_start:idx_end], stddev[idx_start:idx_end])) for name, (idx_start, idx_end) in self.param_idx_ranges.items()])


def _get_base_dist(dist):
    if isinstance(dist, Independent):
        return _get_base_dist(dist.base_dist)
    else:
        return dist

class IllCondException(Exception):
    "Raised when a matrix is ill-conditioned"
    pass
