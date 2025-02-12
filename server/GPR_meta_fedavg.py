import torch, gpytorch, time, warnings
import numpy as np

from server.models import LearnedGPRegressionModel, NeuralNetwork, AffineTransformedDistribution
from server.util import _handle_input_dimensionality, DummyLRScheduler, warning_show, warning_format, WrapLogger, softplus_inverse
from server.abstract import RegressionModelMetaLearned
from config import device

warnings.formatwarning = warning_format
#warnings.showwarning = warning_show

class GPRegressionMetaFedAvg(RegressionModelMetaLearned):

    def __init__(self, clients_train_data, learning_mode='both', feature_dim=2,
                 min_noise_std=1e-3, optimize_noise=True, noise_std=None,
                 lengthscale_fix=None, optimize_lengthscale=True,
                 covar_module_str='NN', mean_module_str='NN', mean_nn_layers=(32, 32), kernel_nn_layers=(32, 32),
                 nonlinearity_hidden_m=torch.tanh, nonlinearity_hidden_k=torch.tanh,
                 nonlinearity_output_m=None, nonlinearity_output_k=None,
                 num_iter_fit=10000, lr=1e-3, lr_decay=1.0, weight_decay=0.0, logger=None,
                 task_batch_size=5, normalize_data=True, optimizer='Adam', random_seed=None, ts_data=False):
        """
        minimizing average log lieklihood over all tasks.
        Note: not a PACOH approximation. A baseline alternative.

        Args:
            clients_train_data: list of tuples of ndarrays[(train_x_1, train_y_1), ..., (train_x_n, train_y_n)]
            learning_mode: (str) specifying which of the GP prior parameters to optimize. Either one of
                    ['learned_mean', 'learned_kernel', 'both', 'vanilla']
            lr: (float) learning rate for AdamW optimizing GP prior hyper-parameters
            lr_decay: (float) multiplicative learning rate decay applied every 1000 iterations.
                      set to 1 to disactivate. acts independently from Adam lr changes.
            weight_decay: (float) l2 regularization of hyper-parameters.
                      set to 0 to deactivate. default in AdamW=0.01
            feature_dim: (int) output dimensionality of NN feature map for kernel function
            num_iter_fit: (int) number of gradient steps for fitting the parameters
            covar_module_str: name of kernel module, default: RBF kernel
            mean_module_str: name of mean module, default: ZeroMean
            mean_nn_layers: (tuple) hidden layer sizes of mean NN
            kernel_nn_layers: (tuple) hidden layer sizes of kernel NN
            task_batch_size: (int) batch size for meta training, i.e. number of tasks for computing gradients
            optimizer: (str) type of optimizer to use - must be either 'Adam' or 'SGD'
            random_seed: (int) seed for pytorch
            noise-related args:
            - optimize_noise: if true, optimizes the noise and applies softplus to enforce the
                              constraint learned_noise_std >= min_noise_std
                              if false, fixes the noise to noise_std
            - noise_std: fixed noise_std, used if optimize_noise==True
            - min_noise_std: constraint on minimum noise std of the Gaussian likelihood, used if optimize_noise==False
        """
        super().__init__(normalize_data, random_seed)

        assert learning_mode in ['learn_mean', 'learn_kernel', 'both', 'vanilla']
        assert mean_module_str in ['NN', 'linear', 'constant', 'zero']
        assert covar_module_str in ['NN', 'SE', 'linear']
        assert optimizer in ['Adam', 'SGD']

        self.lr, self.weight_decay, self.feature_dim = lr, weight_decay, feature_dim
        self.num_iter_fit, self.task_batch_size, self.normalize_data = num_iter_fit, task_batch_size, normalize_data
        self.input_dim = clients_train_data[0][0].shape[1]
        self.ts_data, self.learning_mode = ts_data, learning_mode

        # mean and kernel info
        self.mean_module_str, self.covar_module_str = mean_module_str, covar_module_str
        self.mean_nn_layers, self.kernel_nn_layers = mean_nn_layers, kernel_nn_layers
        self.nonlinearity_hidden_m, self.nonlinearity_hidden_k = nonlinearity_hidden_m, nonlinearity_hidden_k
        self.nonlinearity_output_m, self.nonlinearity_output_k = nonlinearity_output_m, nonlinearity_output_k

        # learning the lengthscale
        if optimize_lengthscale:
            assert lengthscale_fix is None, "fix lengthscale or optimize it, cannot do both."
        else:
            assert not (lengthscale_fix is None), "must provide lengthscale if not optimizing it."
        self.optimize_lengthscale, self.lengthscale_fix = optimize_lengthscale, lengthscale_fix

        # Gaussian likelihood noise
        self.min_noise_std, self.optimize_noise, self.noise_std = min_noise_std, optimize_noise, noise_std
        if self.optimize_noise:
            assert self.noise_std is None, "fix noise or optimize it, cannot do both."
        else:
            assert not (self.noise_std is None), "must provide noise std if not optimizing it."

        # Check that data all has the same size
        self._check_meta_data_shapes(clients_train_data)
        self._compute_normalization_stats(clients_train_data)

        # Setup components that are shared across tasks
        self._setup_gp_prior(
            mean_module_str, covar_module_str, learning_mode, feature_dim,
            mean_nn_layers, kernel_nn_layers, lengthscale_fix,
            nonlinearity_hidden_m, nonlinearity_hidden_k,
            nonlinearity_output_m, nonlinearity_output_k)

        # Likelihood
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.likelihoods.noise_models.GreaterThan(self.min_noise_std))

        if not self.optimize_noise:
            self.likelihood.noise = self.noise_std
            self.likelihood.noise_covar.raw_noise.to(device)
            self.likelihood.noise_covar.raw_noise.requires_grad_(False)
        self.likelihood = self.likelihood.to(device)

        # add likelihood params to params to optimize
        if self.optimize_noise:
            self.shared_parameters.append({'params': self.likelihood.parameters(), 'lr': self.lr})


        # Setup components that are different across tasks
        self.task_dicts = []

        for train_x, train_y in clients_train_data:
            task_dict = {}

            # a) prepare data
            x_tensor, y_tensor = self._prepare_data_per_task(train_x, train_y)
            task_dict['train_x'], task_dict['train_y'] = x_tensor, y_tensor

            # b) prepare model
            task_dict['model'] = LearnedGPRegressionModel(task_dict['train_x'], task_dict['train_y'], self.likelihood,
                                              learned_kernel=self.nn_kernel_map, learned_mean=self.nn_mean_fn,
                                              covar_module=self.covar_module, mean_module=self.mean_module)
            task_dict['mll_fn'] = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, task_dict['model']).to(device)

            self.task_dicts.append(task_dict)

        # c) prepare inference
        self._setup_optimizer(optimizer, lr, lr_decay)

        self.fitted = False
        self.logger=WrapLogger(logger) if not isinstance(logger, WrapLogger) else logger


    def fit(self, clients_data=None, log_period=500):
        """
        meta-learns the GP prior parameters

        Args:
            clients_data: list of clients tuples, i.e. [(test_context_x_1, test_context_t_1, test_x_1, test_t_1), ...]
            log_period: (int) number of steps after which to print stats
        """
        for task_dict in self.task_dicts: task_dict['model'].train()
        self.likelihood.train()

        assert (clients_data is None) or (all([len(data) == 4 for data in clients_data]))

        if len(self.shared_parameters) > 0:
            t = time.time()
            cum_loss = 0.0

            for itr in range(1, self.num_iter_fit + 1):

                loss = 0.0
                self.optimizer.zero_grad()

                for task_dict in self.rds_numpy.choice(self.task_dicts, size=self.task_batch_size):
                    output = task_dict['model'](task_dict['train_x'])
                    mll = task_dict['mll_fn'](output, task_dict['train_y'])
                    loss -= mll

                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                cum_loss += loss

                # print training stats stats
                if itr == 1 or itr % log_period == 0:
                    duration = time.time() - t
                    avg_loss = cum_loss / (log_period if itr > 1 else 1.0)
                    cum_loss = 0.0
                    t = time.time()

                    message = 'Iter %d/%d - Loss: %.6f - Time %.2f sec' % (itr, self.num_iter_fit, avg_loss.item(), duration)

                    # if validation data is provided  -> compute the valid log-likelihood
                    if clients_data is not None:
                        self.likelihood.eval()
                        res = self.eval_datasets(clients_data)
                        valid_nll, valid_rmse, calibr_err = res['nll'], res['rmse'], res['calibr']
                        self.likelihood.train()
                        message += ' - Neg-Valid-LL: %.3f - Valid-RMSE: %.3f - Calib-Err %.3f' % (valid_nll, valid_rmse, calibr_err)

                    self.logger.info(message)

        else:
            self.logger.info('Vanilla mode - nothing to fit')

        self.fitted = True

        for task_dict in self.task_dicts: task_dict['model'].eval()
        self.likelihood.eval()
        return loss.item()


    def predict(self, context_x, context_y, test_x, return_density=False):
        """
        Performs posterior inference (target training) with (context_x, context_y) as training data and then
        computes the predictive distribution of the targets p(y|test_x, test_context_x, context_y) in the test points

        Args:
            context_x: (ndarray) context input data for which to compute the posterior
            context_y: (ndarray) context targets for which to compute the posterior
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            return_density: (bool) whether to return result as mean and std ndarray or as MultivariateNormal pytorch object

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation corresponding to p(t|test_x, test_context_x, context_y)
        """

        context_x, context_y = _handle_input_dimensionality(context_x, context_y)
        test_x = _handle_input_dimensionality(test_x)
        assert test_x.shape[1] == context_x.shape[1]

        # normalize data and convert to tensor
        context_x, context_y = self._prepare_data_per_task(context_x, context_y)

        test_x = self._normalize_data(X=test_x, Y=None)
        test_x = torch.from_numpy(test_x).float().to(device)

        with torch.no_grad():
            # compute posterior given the context data
            gp_model = LearnedGPRegressionModel(context_x, context_y, self.likelihood,
                                                learned_kernel=self.nn_kernel_map, learned_mean=self.nn_mean_fn,
                                                covar_module=self.covar_module, mean_module=self.mean_module)

            gp_model.train()  # to clear the cache
            gp_model.eval()
            self.likelihood.train()  # to clear the cache
            self.likelihood.eval()

            if self.ts_data:
                pred_mean = torch.zeros(test_x.shape[0]).to(device)
                pred_var = torch.zeros((test_x.shape[0], test_x.shape[0])).to(device) # diagonal matrix
                for point_num in np.arange(test_x.shape[0]):
                    if self.optimize_noise:
                        pred_dist = self.likelihood(gp_model(test_x[point_num, :].reshape(1, -1)))
                    else:
                        pred_dist = self.likelihood(
                            gp_model(test_x[point_num, :]),
                            noise=self.noise_std*torch.ones(1))
                    pred_mean[point_num] = pred_dist.mean
                    pred_var[point_num, point_num] = pred_dist.variance
                pred_dist = gpytorch.distributions.MultivariateNormal(pred_mean, pred_var)
            else:
                if self.optimize_noise:
                    pred_dist = self.likelihood(gp_model(test_x))
                else:
                    pred_dist = self.likelihood(
                        gp_model(test_x),
                        noise=self.noise_std*torch.ones(test_x.shape[0]))

            pred_dist_transformed = AffineTransformedDistribution(
                pred_dist, normalization_mean=self.y_mean, normalization_std=self.y_std)

            if return_density:
                return pred_dist_transformed
            else:
                pred_mean = pred_dist_transformed.mean.cpu().numpy()
                pred_std = pred_dist_transformed.stddev.cpu().numpy()
                return pred_mean, pred_std


    def predict_only_mean(self, context_x, context_y, test_x):
        """
        predictions only using GP mean
        """
        context_x, context_y = _handle_input_dimensionality(context_x, context_y)
        test_x = _handle_input_dimensionality(test_x)
        assert test_x.shape[1] == context_x.shape[1]
        # normalize data and convert to tensor
        context_x, context_y = self._prepare_data_per_task(context_x, context_y)
        test_x = self._normalize_data(X=test_x, Y=None)
        test_x = torch.from_numpy(test_x).float().to(device)

        with torch.no_grad():
            # compute posterior given the context data
            gp_model = LearnedGPRegressionModel(context_x, context_y, self.likelihood,
                                                learned_kernel=self.nn_kernel_map, learned_mean=self.nn_mean_fn,
                                                covar_module=self.covar_module, mean_module=self.mean_module)
            gp_model.train()  # to clear the cache
            gp_model.eval()
            pred_mean = gp_model.learned_mean(test_x).squeeze().detach().numpy()
            return pred_mean*self.y_std+self.y_mean


    def serialize_model(self):
        # general properties
        srz_mdl = {
            'min_noise_std':self.min_noise_std, 'optimize_noise':self.optimize_noise,
            'noise_std':self.noise_std, 'optimize_lengthscale':self.optimize_lengthscale,
            'num_iter_fit':self.num_iter_fit, 'lr':self.lr, 'weight_decay':self.weight_decay,
            'task_batch_size':self.task_batch_size, 'normalize_data':self.normalize_data,
            'ts_data':self.ts_data, 'lengthscale_fix':self.lengthscale_fix,
            'learning_mode':self.learning_mode, 'feature_dim': self.feature_dim,
            'mean_module_str':self.mean_module_str, 'covar_module_str':self.covar_module_str,
            'mean_nn_layers':self.mean_nn_layers, 'kernel_nn_layers':self.kernel_nn_layers,
            'nonlinearity_hidden_m':self.nonlinearity_hidden_m,
            'nonlinearity_hidden_k':self.nonlinearity_hidden_k,
            'nonlinearity_output_m':self.nonlinearity_output_m,
            'nonlinearity_output_k':self.nonlinearity_output_k
        }
        # properties NOT saved: lr_decay, optimizer, logger
        # models and optimizer
        srz_mdl['optimizer']=self.optimizer.state_dict()
        srz_mdl['model']=self.task_dicts[0]['model'].state_dict()

        for task_dict in self.task_dicts:
            for key, tensor in task_dict['model'].state_dict().items():
                assert torch.all(srz_mdl['model'][key] == tensor).item()
        return srz_mdl


    def _setup_gp_prior(
        self, mean_module_str, covar_module_str, learning_mode,
        feature_dim, mean_nn_layers, kernel_nn_layers, lengthscale_fix,
        nonlinearity_hidden_m, nonlinearity_hidden_k,
        nonlinearity_output_m, nonlinearity_output_k):

        self.shared_parameters = []

        # a) determine kernel map & module
        if covar_module_str == 'NN':
            assert learning_mode in ['learn_kernel', 'both'], 'neural network parameters must be learned'
            self.nn_kernel_map = NeuralNetwork(
                input_dim=self.input_dim, output_dim=feature_dim,
                layer_sizes=kernel_nn_layers, requires_bias={'hidden':True, 'out':False},
                nonlinearity_hidden=nonlinearity_hidden_k, nonlinearity_output=nonlinearity_output_k
                ).to(device)
            self.shared_parameters.append(
                {'params': self.nn_kernel_map.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay})
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim)).to(device)
            if not self.optimize_lengthscale:
                self.covar_module.base_kernel.lengthscale_raw = torch.tensor(softplus_inverse(lengthscale_fix)).float().to(device)
                self.covar_module.base_kernel.lengthscale_raw.requires_grad=False
        else:
            self.nn_kernel_map = None

        if covar_module_str == 'SE':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=self.input_dim)).to(device)
            if not self.optimize_lengthscale:
                self.covar_module.base_kernel.lengthscale = torch.tensor(lengthscale_fix).float().to(device)
                self.covar_module.base_kernel.lengthscale.requires_grad=False
        elif covar_module_str == 'linear':
            self.covar_module = gpytorch.kernels.LinearKernel(num_dimensions=self.input_dim).to(device)
        # elif isinstance(covar_module, gpytorch.kernels.Kernel):
        #     self.covar_module = covar_module.to(device)

        # b) determine mean map & module

        if mean_module_str == 'NN':
            assert learning_mode in ['learn_mean', 'both'], 'neural network parameters must be learned'
            self.nn_mean_fn = NeuralNetwork(
                input_dim=self.input_dim, output_dim=1, layer_sizes=mean_nn_layers,
                requires_bias={'hidden':True, 'out':True},
                nonlinearity_hidden=nonlinearity_hidden_m, nonlinearity_output=nonlinearity_output_m
            ).to(device)
            self.shared_parameters.append(
                {'params': self.nn_mean_fn.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay})
            self.mean_module = None
        else:
            self.nn_mean_fn = None

        if mean_module_str == 'constant':
            self.mean_module = gpytorch.means.ConstantMean().to(device)
        elif mean_module_str == 'zero':
            self.mean_module = gpytorch.means.ZeroMean().to(device)
        elif mean_module_str == 'linear':
            self.mean_module = gpytorch.means.LinearMean(input_size=self.input_dim).to(device)
        # elif isinstance(mean_module, gpytorch.means.Mean):
        #     self.mean_module = mean_module.to(device)

        # c) add parameters of covar and mean module if desired

        if learning_mode in ["learn_kernel", "both"]:
            self.shared_parameters.append({'params': self.covar_module.hyperparameters(), 'lr': self.lr})

        if learning_mode in ["learn_mean", "both"] and self.mean_module is not None:
            self.shared_parameters.append({'params': self.mean_module.hyperparameters(), 'lr': self.lr})

    def _setup_optimizer(self, optimizer, lr, lr_decay):
        if optimizer == 'Adam':
            self.optimizer = torch.optim.AdamW(self.shared_parameters, lr=lr, weight_decay=self.weight_decay)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.shared_parameters, lr=lr)
        else:
            raise NotImplementedError('Optimizer must be Adam or SGD')

        if lr_decay < 1.0:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1000, gamma=lr_decay)
            # Multiplicative decay of each parameter group lr by gamma every step_size epochs.
            # decay can happen simultaneously with other changes to the lr from outside this scheduler.
        else:
            self.lr_scheduler = DummyLRScheduler()

    def _vectorize_pred_dist(self, pred_dist):
        return torch.distributions.Normal(pred_dist.mean, pred_dist.stddev)


def load_serialized_fedavg_model(clients_train_data, srz_mdl):
    # construct a model with a subset of props which were saved
    gp = GPRegressionMetaFedAvg(
        clients_train_data=clients_train_data,
        min_noise_std=srz_mdl['min_noise_std'], optimize_noise=srz_mdl['optimize_noise'],
        noise_std=srz_mdl['noise_std'], optimize_lengthscale=srz_mdl['optimize_lengthscale'],
        num_iter_fit=srz_mdl['num_iter_fit'], lr=srz_mdl['lr'], weight_decay=0.0,
        task_batch_size=srz_mdl['task_batch_size'], normalize_data=srz_mdl['normalize_data'],
        ts_data=srz_mdl['ts_data'], lengthscale_fix=srz_mdl['lengthscale_fix'],
        learning_mode=srz_mdl['learning_mode'], feature_dim=srz_mdl['feature_dim'],
        mean_module_str=srz_mdl['mean_module_str'], covar_module_str=srz_mdl['covar_module_str'],
        mean_nn_layers=srz_mdl['mean_nn_layers'], kernel_nn_layers=srz_mdl['kernel_nn_layers'],
        nonlinearity_hidden_m=srz_mdl['nonlinearity_hidden_m'],
        nonlinearity_hidden_k=srz_mdl['nonlinearity_hidden_k'],
        nonlinearity_output_m=srz_mdl['nonlinearity_output_m'],
        nonlinearity_output_k=srz_mdl['nonlinearity_output_k']
        )
    # replace default models with learned models
    for task_dict in gp.task_dicts:
        task_dict['model'].load_state_dict(srz_mdl['model'])
    # load state dict for optimizer
    gp.optimizer.load_state_dict(srz_mdl['optimizer'])
    return gp




if __name__ == "__main__":
    from experiments.data_sim import GPFunctionsDataset, SinusoidDataset

    clients_data = SinusoidDataset(random_state=np.random.RandomState(29))
    clients_train_data = data_sim.generate_clients_train_data(n_tasks=20, n_samples=10)
    #meta_test_data = data_sim.generate_meta_test_data(n_tasks=50, n_samples_context=10, n_samples_test=160)

    NN_LAYERS = (32, 32, 32, 32)

    plot = False
    from matplotlib import pyplot as plt

    if plot:
        for x_train, y_train in clients_train_data:
            plt.scatter(x_train, y_train)
        plt.title('sample from the GP prior')
        plt.show()

    """ 2) Classical mean learning based on mll """

    print('\n ---- GPR mll meta-learning ---- ')

    torch.set_num_threads(2)

    for weight_decay in [0.8, 0.5, 0.4, 0.3, 0.2, 0.1]:
        gp_model = GPRegressionMetaFedAvg(clients_train_data, num_iter_fit=20000, weight_decay=weight_decay, task_batch_size=2,
                                             covar_module='NN', mean_module='NN', mean_nn_layers=NN_LAYERS,
                                             kernel_nn_layers=NN_LAYERS)
        itrs = 0
        print("---- weight-decay =  %.4f ----"%weight_decay)
        for i in range(1):
            gp_model.fit(clients_data=clients_data, log_period=1000)
            itrs += 20000

            x_plot = np.linspace(-5, 5, num=150)
            x_context, t_context, x_test, y_test = meta_test_data[0]
            pred_mean, pred_std = gp_model.predict(x_context, t_context, x_plot)
            ucb, lcb = gp_model.confidence_intervals(x_context, t_context, x_plot, confidence=0.9)

            plt.scatter(x_test, y_test)
            plt.scatter(x_context, t_context)

            plt.plot(x_plot, pred_mean)
            plt.fill_between(x_plot, lcb, ucb, alpha=0.2)
            plt.title('GPR meta mll (weight-decay =  %.4f) itrs = %i' % (weight_decay, itrs))
            plt.show()