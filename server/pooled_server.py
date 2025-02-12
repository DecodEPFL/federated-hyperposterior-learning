import torch, gpytorch, copy
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from config import device
from server.abstract import _calib_error
from server.models import NeuralNetwork
from server.util import softplus_inverse

class PooledGPModel(ApproximateGP):
    def __init__(
        self, inducing_points, lr=1e-2, weight_decay=0.0,
        mean_module_str='constant', covar_module_str='SE',
        feature_dim=2, kernel_nn_layers=None, mean_nn_layers=None,
        lengthscale_fix=None, optimize_lengthscale=True,
        nonlinearity_hidden_m=torch.tanh, nonlinearity_output_m=None,
        nonlinearity_hidden_k=torch.tanh, nonlinearity_output_k=None):

        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(PooledGPModel, self).__init__(variational_strategy)
        self.lr, self.weight_decay = lr, weight_decay
        self.shared_parameters = []
        self.input_dim = inducing_points.shape[1]
        self.optimize_lengthscale = optimize_lengthscale

        # a) determine kernel map & module
        if covar_module_str == 'NN':
            self.nn_kernel_map = NeuralNetwork(
                input_dim=self.input_dim, output_dim=feature_dim,
                layer_sizes=kernel_nn_layers, requires_bias={'hidden':True, 'out':False},
                nonlinearity_hidden=nonlinearity_hidden_k,
                nonlinearity_output=nonlinearity_output_k
            ).to(device)
            self.shared_parameters.append(
                {'params': self.nn_kernel_map.parameters(),
                'lr': self.lr, 'weight_decay': self.weight_decay}
            )
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim)
            ).to(device)
            if not self.optimize_lengthscale:
                self.covar_module.base_kernel.lengthscale_raw = torch.tensor(
                    softplus_inverse(lengthscale_fix)
                ).float().to(device)
                self.covar_module.base_kernel.lengthscale_raw.requires_grad=False
        else:
            self.nn_kernel_map = None

        if covar_module_str == 'SE':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=self.input_dim)
            ).to(device)
            if not self.optimize_lengthscale:
                self.covar_module.base_kernel.lengthscale = torch.tensor(
                    lengthscale_fix
                ).float().to(device)
                self.covar_module.base_kernel.lengthscale.requires_grad=False
        elif covar_module_str == 'linear':
            self.covar_module = gpytorch.kernels.LinearKernel(
                num_dimensions=self.input_dim
            ).to(device)


        # b) determine mean map & module
        if mean_module_str == 'NN':
            self.nn_mean_fn = NeuralNetwork(
                input_dim=self.input_dim, output_dim=1, layer_sizes=mean_nn_layers,
                requires_bias={'hidden':True, 'out':True},
                nonlinearity_hidden=nonlinearity_hidden_m,
                nonlinearity_output=nonlinearity_output_m
            ).to(device)
            self.shared_parameters.append(
                {'params': self.nn_mean_fn.parameters(),
                'lr': self.lr, 'weight_decay': self.weight_decay})
            self.mean_module = None
        else:
            self.nn_mean_fn = None

        if mean_module_str == 'constant':
            self.mean_module = gpytorch.means.ConstantMean().to(device)
        elif mean_module_str == 'zero':
            self.mean_module = gpytorch.means.ZeroMean().to(device)
        elif mean_module_str == 'linear':
            self.mean_module = gpytorch.means.LinearMean(
                input_size=self.input_dim
            ).to(device)

        # c) add parameters of covar and mean module
        if self.covar_module is not None:
            self.shared_parameters.append({'params': self.covar_module.hyperparameters(),
            'lr': self.lr})
        if self.mean_module is not None:
            self.shared_parameters.append({'params': self.mean_module.hyperparameters(),
            'lr': self.lr})


    def forward(self, x):
        # feed through kernel NN
        if self.nn_kernel_map is not None:
            projected_x = self.nn_kernel_map(x)
        else:
            projected_x = x
        covar_x = self.covar_module(projected_x)
        # feed through mean module
        if self.nn_mean_fn is not None: # mean is NN
            mean_x = self.nn_mean_fn(x).squeeze()
        else:
            mean_x = self.mean_module(x).squeeze()
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    # def eval(self, loader):
    #     """
    #     Performs posterior inference over all batches in loader, and
    #     computes the average test log likelihood, rmse and calibration error

    #     Args:
    #         loader: data loader

    #     Returns: (neg avg_log_likelihood, rmse, calibr_error)

    #     """
    #     self.eval()


    #     x_train, y_train = _handle_input_dimensionality(x_train, y_train)
    #     x_valid, y_valid = _handle_input_dimensionality(x_valid, y_valid)
    #     if flatten_y:
    #         y_valid_tensor = torch.from_numpy(y_valid).float().flatten().to(device)
    #     else:
    #         y_valid_tensor = torch.unsqueeze(torch.from_numpy(y_valid).float().to(device), dim=0)

    #     pred_dist = self.predict(x_train, y_train, x_valid, return_density=True, **kwargs)
    #     """ pred_dist is class 'server.models.EqualWeightedMixtureDist'
    #         pred_dist.mean and pred_dist.variance \in R^{num_test_samples}
    #         note: variance is a vector, b.c. pred_dist is stack of Gaussians at test points
    #         GP predictive dist has a full cov matrix, but here we only need the diagonal values.
    #         pred_dist.dists is AffineTransformedDistribution(), affine mix of posteriors corresponding to different particles """

    #     avg_log_likelihood = torch.mean(pred_dist.log_prob(y_valid_tensor) / y_valid_tensor.shape[0])
    #     rmse = torch.mean(torch.pow(pred_dist.mean - y_valid_tensor, 2)).sqrt()

    #     pred_dist_vect = self._vectorize_pred_dist(pred_dist)
    #     calibr_error = self._calib_error(pred_dist_vect, y_valid_tensor)

    #     return -1*avg_log_likelihood.cpu().item(), rmse.cpu().item(), calibr_error.cpu().item()



    def state_dict(self):
        state_dict = {
            'model': copy.deepcopy(self.model.state_dict()),
            'optimizer': copy.deepcopy(self.optimizer.state_dict())
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def _vectorize_pred_dist(self, pred_dist):
        return torch.distributions.Normal(pred_dist.mean, pred_dist.stddev)