import gpytorch, time, torch, copy, warnings
import numpy as np

from config import device
from server.models import LearnedGPRegressionModel, NeuralNetwork, AffineTransformedDistribution
from server.abstract import RegressionModel
from server.util import _handle_input_dimensionality, warning_show, warning_format, softplus_inverse

#warnings.showwarning = warning_show
warnings.formatwarning = warning_format

# change gpytorch settings to avoid non-PSD errors
gpytorch.settings.cholesky_jitter(float=1e-4, double=1e-6)
# Default for float: 1e-6, Default for double: 1e-8
gpytorch.settings.cholesky_max_tries(value=8) # default = 3


class GPRegressionLearned(RegressionModel):

    def __init__(self, train_x, train_t, learning_mode='both', lr=1e-3, weight_decay=0.0, feature_dim=2, min_noise_std=1e-3,
                 noise_std=None, optimize_noise=True, lengthscale_fix=None, optimize_lengthscale=True, early_stopping=True,
                 num_iter_fit=1000, covar_module='NN', mean_module='NN', mean_nn_layers=(32, 32), kernel_nn_layers=(32, 32),
                 nonlinearity_hidden_m=torch.tanh, nonlinearity_hidden_k=torch.tanh,
                 nonlinearity_output_m=None, nonlinearity_output_k=torch.tanh,
                 optimizer='Adam', normalize_data=True, lr_scheduler=True, random_seed=None, ts_data=False):
        """
        Gaussian Process Regression with learnable mean and kernel function.
        Note that this class does not perform any meta-learning. The GP priors mean and kernel function are learned
        based on the same train dataset that is also used for posterior inference.

        Args:
            train_x: (ndarray) train inputs - shape: (n_samples, ndim_x)
            train_t: (ndarray) train targets - shape: (n_samples, 1)
            learning_mode: (str) specifying which of the GP prior parameters to optimize. Either one of
                    ['learned_mean', 'learned_kernel', 'both', 'vanilla']
            lr: (float) learning rate for prior parameters
            weight_decay: (float) weight decay penalty
            lr_scheduler: use ReduceLROnPlateau
            early_stopping: (boolean) if True, returns the model with the lowest validation RMSE.
                            models are only evaluated after log_period.
                            NOTE: continues training, but returns the best one seen at an earlier iteration
            feature_dim: (int) output dimensionality of NN feature map for kernel function
            num_iter_fit: (int) number of gradient steps for fitting the parameters
            covar_module: (gpytorch.mean.Kernel) optional kernel module, default: RBF kernel
            mean_module: (gpytorch.mean.Mean) optional mean module, default: ZeroMean
            mean_nn_layers: (tuple) hidden layer sizes of mean NN
            kernel_nn_layers: (tuple) hidden layer sizes of kernel NN
            optimizer: (str) type of optimizer to use - must be either 'Adam' or 'SGD'
            random_seed: (int) seed for pytorch

            noise-related args:
            - optimize_noise: if true, optimizes the noise and applies softplus to enforce the
                              constraint learned_noise_std >= min_noise_std
                              if flase, fixes the noise to noise_std (sim for lengthscale)
            - noise_std: fixed noise_std, used if optimize_noise==True (sim for lengthscale)
            - min_noise_std: constraint on the learned noise std, used if optimize_noise==False
        """
        super().__init__(normalize_data=normalize_data, random_seed=random_seed)

        assert learning_mode in ['learn_mean', 'learn_kernel', 'both', 'vanilla']
        assert mean_module in ['NN', 'constant', 'zero', 'linear'] or isinstance(mean_module, gpytorch.means.Mean)
        assert covar_module in ['NN', 'SE', 'linear', 'periodic'] or isinstance(covar_module, gpytorch.kernels.Kernel)
        assert optimizer in ['Adam', 'SGD']

        self.lr, self.weight_decay, self.num_iter_fit, self.lr_scheduler = lr, weight_decay, num_iter_fit, lr_scheduler
        self.early_stopping = early_stopping
        self.ts_data = ts_data

        # learning the lengthscale
        if optimize_lengthscale:
            assert lengthscale_fix is None, "fix lengthscale or optimize it, cannot do both."
        else:
            assert not (lengthscale_fix is None), "must provide lengthscale if not optimizing it."
        self.optimize_lengthscale = optimize_lengthscale

        # Gaussian likelihood noise
        self.min_noise_std, self.optimize_noise, self.noise_std = min_noise_std, optimize_noise, noise_std
        if self.optimize_noise:
            assert self.noise_std is None, "fix noise or optimize it, cannot do both."
        else:
            assert not (self.noise_std is None), "must provide noise std if not optimizing it."

        """ ------ Data handling ------ """
        self.train_x_tensor, self.train_t_tensor = self._initial_data_handling(train_x, train_t)
        assert self.train_t_tensor.shape[-1] == 1
        self.train_t_tensor = self.train_t_tensor.flatten()

        """  ------ Setup model ------ """
        self.parameters = []
        self.fitted = False

        # A) determine kernel map & module
        nn_kernel_map = None
        if covar_module == 'NN':
            assert learning_mode in ['learn_kernel', 'both'], 'neural network parameters must be learned'
            nn_kernel_map = NeuralNetwork(
                                input_dim=self.input_dim, output_dim=feature_dim,
                                layer_sizes=kernel_nn_layers,
                                nonlinearity_hidden=nonlinearity_hidden_k,
                                nonlinearity_output=nonlinearity_output_k,
                                requires_bias={'hidden':True, 'out':False}).to(device)
            self.parameters.append({'params': nn_kernel_map.parameters(), 'lr': self.lr,
                                    'weight_decay': self.weight_decay})
            covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(
                                                ard_num_dims=feature_dim)).to(device)
            if not self.optimize_lengthscale:
                covar_module.base_kernel.lengthscale = torch.tensor(softplus_inverse(lengthscale_fix)).float().to(device)
                covar_module.base_kernel.lengthscale.requires_grad=False
        elif covar_module == 'SE':
            covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(
                                            ard_num_dims=self.input_dim)).to(device)
            if not self.optimize_lengthscale:
                covar_module.base_kernel.lengthscale = torch.tensor(softplus_inverse(lengthscale_fix)).float().to(device)
                covar_module.base_kernel.lengthscale.requires_grad=False
        elif covar_module == 'linear':
            covar_module = LinearKernelLight(num_dimensions=self.input_dim).to(device)
            #covar_module = gpytorch.kernels.LinearKernel(num_dimensions=self.input_dim).to(device)
        elif covar_module == 'periodic':
            covar_module = gpytorch.kernels.PeriodicKernel(ard_num_dims=train_x.shape[1]).to(device) #TODO: double check the size
            # learns separate lengthscale for each input dimension
        else:
            covar_module = covar_module.to(device)



        # B) determine mean map & module
        if mean_module == 'NN':
            assert learning_mode in ['learn_mean', 'both'], 'neural network parameters must be learned'
            nn_mean_fn = NeuralNetwork(
                input_dim=self.input_dim, output_dim=1, layer_sizes=mean_nn_layers,
                nonlinearity_hidden=nonlinearity_hidden_m,
                nonlinearity_output=nonlinearity_output_m,
                requires_bias={'hidden':True, 'out':True}).to(device)
            self.parameters.append({'params': nn_mean_fn.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay})
            mean_module = None
        else:
            nn_mean_fn = None

        if mean_module == 'constant':
            mean_module = gpytorch.means.ConstantMean().to(device)
        elif mean_module == 'zero':
            mean_module = gpytorch.means.ZeroMean().to(device)
        elif mean_module == 'linear':
            mean_module = gpytorch.means.LinearMean(input_size=self.input_dim, bias=True).to(device)
        elif not mean_module is None: # if mean is an instance of gpytorch.means
            mean_module = mean_module.to(device)

        # C) setup GP model
        if self.optimize_noise:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.likelihoods.noise_models.GreaterThan(self.min_noise_std)
                ).to(device)
        else:
            learn_additional_noise = False # TODO: set to True
            noises = torch.ones(self.train_x_tensor.shape[0]) * self.noise_std
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=noises, learn_additional_noise=learn_additional_noise).to(device)

        if self.optimize_noise or ((not self.optimize_noise) and learn_additional_noise):
            self.parameters.append({'params': self.likelihood.parameters(), 'lr': self.lr})


        self.model = LearnedGPRegressionModel(self.train_x_tensor, self.train_t_tensor, self.likelihood,
                                              learned_kernel=nn_kernel_map, learned_mean=nn_mean_fn,
                                              covar_module=covar_module, mean_module=mean_module)

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).to(device)


        # D) determine which parameters are trained and setup optimizer

        if learning_mode in ["learn_kernel", "both"]:
            self.parameters.append({'params': self.model.covar_module.hyperparameters(), 'lr': self.lr})

        if learning_mode in ["learn_mean", "both"] and mean_module is not None:
            self.parameters.append({'params': self.model.mean_module.hyperparameters(), 'lr': self.lr})

        if optimizer == 'Adam':
            self.optimizer = torch.optim.AdamW(self.parameters)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters)
        else:
            raise NotImplementedError('Optimizer must be Adam or SGD')

        if self.lr_scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.2)
        else:  # factor 1.0 --> no lr decay
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=1.0)



    def fit(self, valid_x=None, valid_t=None, verbose=True, log_period=500):
        """
        fits GP prior parameters of by maximizing the marginal log-likelihood (mll) of the training data

        Args:
            verbose: (boolean) whether to print training progress
            valid_x: (np.ndarray) validation inputs - shape: (n_samples, ndim_x)
            valid_y: (np.ndarray) validation targets - shape: (n_samples, 1)
            log_period: (int) number of steps after which to print stats
            n_iter: (int) number of gradient descent iterations
        """
        self.model.train()
        self.likelihood.train()

        assert (valid_x is None and valid_t is None) or (isinstance(valid_x, np.ndarray) and isinstance(valid_t, np.ndarray))

        if self.early_stopping:
            valid_rmses, state_dicts = [], []

        if len(self.parameters) > 0:
            t = time.time()
            for itr in range(1, self.num_iter_fit + 1):

                self.optimizer.zero_grad()
                output = self.model(self.train_x_tensor)
                loss = -self.mll(output, self.train_t_tensor)
                loss.backward()
                self.optimizer.step()

                # print training stats
                if itr == 1 or itr % log_period == 0:
                    duration = time.time() - t
                    t = time.time()
                    message = 'Iter %d/%d - Loss: %.3f - Time %.3f sec' % (itr, self.num_iter_fit, loss.item(), duration)

                    # if validation data is provided  -> compute the valid log-likelihood
                    if valid_x is not None:
                        self.model.eval() # enter eval mode
                        self.likelihood.eval()
                        # validation errors
                        valid_ll, valid_rmse, calibr_err = self.eval(valid_x, valid_t)
                        # TODO: this line results in Cholesky error. the following lines were added to solve this
                        # the noise keeps increasing, but the cov still remains non-PSD. maybe the change does not
                        # affect the cov because we're in the eval mode, or maybe the matrix that is inversed doesn't
                        # contain noise
                        '''
                        not_psd_error = True # to enter while
                        while not_psd_error:
                            try:
                                valid_ll, valid_rmse, calibr_err = self.eval(valid_x, valid_t)
                                not_psd_error = False
                            except: # not PSD exception in cholesky
                                print(self.model.likelihood.raw_noise)
                                self.model.train() # enter eval mode
                                self.likelihood.train()
                                self.model.likelihood.raw_noise = 5e-3 + self.model.likelihood.raw_noise # increase raw noise
                                self.model.eval() # enter eval mode
                                self.likelihood.eval()
                                print(self.model.likelihood.raw_noise)
                                not_psd_error = True
                                print('[INFO] increased noise std to resolve non-PSD covariance')
                        '''

                        message += ' - Valid-LL: %.3f - Valid-RMSE: %.3f - Calibration-Err %.3f ' % (valid_ll, valid_rmse, calibr_err)

                        # record parameters
                        if self.early_stopping and itr>1: # skip first iteration
                            valid_rmses.append(valid_rmse)
                            state_dicts.append(self.state_dict())
                        self.lr_scheduler.step(valid_ll)      # update learning rate according to validation error
                        self.model.train()                    # for learning rate to take effect
                        self.likelihood.train()

                    if verbose:
                        self.logger.info(message)
        else:
            self.logger.info('Vanilla mode - nothing to fit')

        # --- early stoping ---
        if self.early_stopping:
            best_ind = valid_rmses.index(min(valid_rmses))
            self.num_iter_fit = (1+best_ind)*log_period     # stop at the best number of iterations
            self.load_state_dict(state_dicts[best_ind])     # put back best parameters
            loss = -self.mll(self.model(self.train_x_tensor), self.train_t_tensor) # recompute loss
        # end of training
        self.fitted = True
        self.model.eval()
        self.likelihood.eval()
        return loss.item()


    def predict(self, test_x, return_density=False, **kwargs):
        """
        computes the predictive distribution of the targets p(t|test_x, train_x, train_y)

        Args:
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            return_density (bool) whether to return a density object or

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation corresponding to p(y_test|X_test, X_train, y_train)
        """

        #if test_x.ndim == 1:
        #    test_x = np.expand_dims(test_x, axis=-1)
        test_x = _handle_input_dimensionality(test_x)
        test_x = self._normalize_data(X=test_x, Y=None)         # normalize data
        test_x = torch.from_numpy(test_x).float().contiguous().to(device)    # convert to tensor ?
        if self.ts_data:
            pred_mean = torch.zeros(test_x.shape[0])
            pred_std = torch.zeros((test_x.shape[0], test_x.shape[0])) # diagonal matrix

        with torch.no_grad():
            if self.ts_data:
                for point_num in np.arange(test_x.shape[0]):
                    if self.optimize_noise:
                        pred_dist = self.likelihood(self.model(test_x[point_num, :]))
                    else:
                        pred_dist = self.likelihood(
                            self.model(test_x[point_num, :]),
                            noise=self.noise_std*torch.ones(1))
                    pred_mean[point_num] = pred_dist.mean
                    pred_std[point_num, point_num] = pred_dist.stddev
            else:
                if self.optimize_noise:
                    pred_dist = self.likelihood(self.model(test_x))
                else:
                    pred_dist = self.likelihood(
                        self.model(test_x),
                        noise=self.noise_std*torch.ones(test_x.shape[0]))
                pred_dist_transformed = AffineTransformedDistribution(
                    pred_dist, normalization_mean=self.y_mean, normalization_std=self.y_std)

            if return_density:
                if self.ts_data:
                    raise NotImplementedError
                return pred_dist_transformed
            else:
                if self.ts_data:
                    return pred_mean.cpu().numpy(), pred_std.cpu().numpy()
                pred_mean = pred_dist_transformed.mean.cpu().numpy()
                pred_std = pred_dist_transformed.stddev.cpu().numpy()
                return pred_mean, pred_std

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

if __name__ == "__main__":
    import torch
    import numpy as np
    from matplotlib import pyplot as plt

    n_train_samples = 20
    n_test_samples = 200

    torch.manual_seed(25)
    x_data = torch.normal(mean=-1, std=2.0, size=(n_train_samples + n_test_samples, 1))
    W = torch.tensor([[0.6]]).float()
    b = torch.tensor([-1]).float()
    y_data = x_data.matmul(W.T) + torch.sin((0.6 * x_data)**2) + b + torch.normal(mean=0.0, std=0.1, size=(n_train_samples + n_test_samples, 1))

    x_data_train, x_data_test = x_data[:n_train_samples].numpy(), x_data[n_train_samples:].numpy()
    y_data_train, y_data_test = y_data[:n_train_samples].numpy(), y_data[n_train_samples:].numpy()

    gp_mll = GPRegressionLearned(
        x_data_train, y_data_train, mean_module='NN', covar_module='SE',
        mean_nn_layers=(32, 32, 32, 32), weight_decay=0.5, num_iter_fit=10000)
    gp_mll.fit(x_data_test, y_data_test)


    x_plot = np.linspace(6, -6, num=200)
    gp_mll.confidence_intervals(x_plot)

    pred_mean, pred_std = gp_mll.predict(x_plot)
    pred_mean, pred_std = pred_mean.flatten(), pred_std.flatten()

    plt.scatter(x_data_test, y_data_test)
    plt.plot(x_plot, pred_mean)

    #lcb, ucb = pred_mean - pred_std, pred_mean + pred_std
    lcb, ucb = gp_mll.confidence_intervals(x_plot)
    plt.fill_between(x_plot, lcb, ucb, alpha=0.4)
    plt.show()
