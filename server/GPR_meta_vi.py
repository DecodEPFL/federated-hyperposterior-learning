import copy, gpytorch, torch, time
import numpy as np

from torch.distributions.multivariate_normal import MultivariateNormal

from server.models import AffineTransformedDistribution, EqualWeightedMixtureDist
from server.hyper_posterior import HyperPosterior, RandomGPPosterior
from server.util import _handle_input_dimensionality, DummyLRScheduler, WrapLogger
from server.abstract import RegressionModelMetaLearned
from config import device

class GPRegressionMetaLearnedVI(RegressionModelMetaLearned):

    def __init__(
        self, meta_train_data, num_iter_fit=10000, feature_dim=1,
        prior_factor=0.01, hyper_prior_dict={},
        covar_module_str='NN', mean_module_str='NN', mean_nn_layers=(32, 32), kernel_nn_layers=(32, 32),
        nonlinearity_hidden_m=torch.tanh, nonlinearity_hidden_k=torch.tanh,
        nonlinearity_output_m=None, nonlinearity_output_k=None,
        likelihood_str='Gaussian', optimize_noise=True, noise_std=None,
        optimize_lengthscale=True, lengthscale_fix=None,
        optimizer='Adam', lr=1e-3, lr_decay=1.0, svi_batch_size=10, cov_type='diag',
        task_batch_size=-1, normalize_data=True, logger=None, random_seed=None, ts_data=False):
        """
        PACOH-VI: Variational Inference on the PAC-optimal hyper-posterior with Gaussian family.
        Meta-Learns a distribution over GP-priors.

        Args:
            meta_train_data: list of tuples of ndarrays[(train_x_1, train_t_1), ..., (train_x_n, train_t_n)]
            num_iter_fit: (int) number of gradient steps for fitting the parameters
            feature_dim: (int) output dimensionality of NN feature map for kernel function
            prior_factor: (float) weighting of the hyper-prior (--> meta-regularization parameter)
            hyper_prior_dict (float): dictionary pf std of Gaussian hyper-prior on weights, biases, lengthscale, noise
            covar_module_str: (gpytorch.mean.Kernel) optional kernel module, default: RBF kernel
            mean_module_str: (gpytorch.mean.Mean) optional mean module, default: ZeroMean
            mean_nn_layers: (tuple) hidden layer sizes of mean NN
            kernel_nn_layers: (tuple) hidden layer sizes of kernel NN
            optimizer: (str) type of optimizer to use - must be either 'Adam' or 'SGD'
            lr: (float) learning rate for prior parameters
            lr_decay: (float) lr rate decay multiplier applied after every 1000 steps
            kernel (std): SVGD kernel, either 'RBF' or 'IMQ'
            bandwidth (float): bandwidth of kernel, if None the bandwidth is chosen via heuristic
            svi_batch_size: number of variational distrinutions per dimension
            task_batch_size: (int) mini-batch size of tasks for estimating gradients
            normalize_data: (bool) whether the data should be normalized
            random_seed: (int) seed for pytorch
            cov_type: cov_type for the VI approximation RandomGPPosterior
        """
        super().__init__(normalize_data, random_seed)

        assert optimizer in ['Adam', 'SGD']
        if not optimize_noise and noise_std is None:
            print('Error: must specify noise std when not optimized')
        self.optimize_noise = optimize_noise
        self.noise_std=noise_std
        self.optimize_lengthscale, self.lengthscale_fix = optimize_lengthscale, lengthscale_fix
        self.num_iter_fit, self.prior_factor, self.feature_dim = num_iter_fit, prior_factor, feature_dim
        self.hyper_prior_dict=hyper_prior_dict
        self.svi_batch_size = svi_batch_size
        if task_batch_size < 1:
            self.task_batch_size = len(meta_train_data)
        else:
            self.task_batch_size = min(task_batch_size, len(meta_train_data))
        self.logger=WrapLogger(logger) if not isinstance(logger, WrapLogger) else logger
        self.ts_data = ts_data

        # Check that data all has the same size
        self._check_meta_data_shapes(meta_train_data)
        self._compute_normalization_stats(meta_train_data)

        """ --- Setup model & inference --- """
        self._setup_model_inference(
            mean_module_str, covar_module_str, mean_nn_layers, kernel_nn_layers,
            nonlinearity_hidden_m, nonlinearity_hidden_k, nonlinearity_output_m, nonlinearity_output_k,
            likelihood_str, cov_type)

        self._setup_optimizer(optimizer, lr, lr_decay)

        # Setup components that are different across tasks
        self.task_dicts = []

        for train_x, train_y in meta_train_data:
            task_dict = {}

            # a) prepare data
            x_tensor, y_tensor = self._prepare_data_per_task(train_x, train_y)
            task_dict['train_x'], task_dict['train_y'] = x_tensor, y_tensor
            self.task_dicts.append(task_dict)

        self.fitted = False

    def meta_fit(
        self, valid_tuples=None, log_period=500, max_iter_fit=None,
        criteria='rsmse', early_stopping=True, record_params=True, cont_fit_margin=None):
        """
        fits the variational hyper-posterior by minimizing the negative ELBO

        Args:
            valid_tuples: list of valid tuples, i.e. [(test_context_x_1, test_context_t_1, test_x_1, test_t_1), ...]
            verbose: (boolean) whether to print training progress
            log_period (int) number of steps after which to print stats
            max_iter_fit: (int) number of gradient descent iterations
        """
        if not cont_fit_margin is None:
            assert NotImplementedError

        if not isinstance(criteria, list):
            criteria = [criteria]
        criteria = [criterion.lower() for criterion in criteria]
        for criterion in criteria:
            assert criterion in ['rmse', 'rsmse', 'calibr', 'nll']

        valid_results = dict.fromkeys(criteria)
        for criterion in criteria:
            valid_results[criterion] = []

        if max_iter_fit is None:
            max_iter_fit = self.num_iter_fit

        num_params = list(self.optimizer.param_groups[0]['params'][0].shape)[0]
        history={'loss': np.empty([max_iter_fit,1]),
                 'map_params': np.empty([max_iter_fit, num_params]),
                 'prior_factor': self.prior_factor}

        if early_stopping:
            self.best_posterior = dict.fromkeys(criteria, None) # track the best posterior
            min_criterion = dict.fromkeys(criteria, 1e6)

        assert (valid_tuples is None) or (all([len(valid_tuple) == 4 for valid_tuple in valid_tuples]))

        t = time.time()

        for itr in range(1, max_iter_fit + 1):
            task_dict_batch = self.rds_numpy.choice(self.task_dicts, size=self.task_batch_size)
            self.optimizer.zero_grad()
            loss = self.get_neg_elbo(task_dict_batch)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            # save to history
            if record_params:
                history['loss'][itr-1] = loss.item()
                history['map_params'][itr-1, :] = copy.deepcopy(self.posterior.mode).cpu().detach().numpy().flatten()

            # print training stats stats
            if itr == 1 or itr % log_period == 0:
                duration = time.time() - t
                t = time.time()

                message = 'Iter %d/%d - Loss: %.6f - Time %.2f sec' % (itr, self.num_iter_fit, loss.item(), duration)

                # if validation data is provided  -> compute the valid log-likelihood
                if valid_tuples is not None:
                    # evaluate on validation set
                    try:
                        valid_res = self.eval_datasets(valid_tuples)
                        for key in valid_results.keys():
                            valid_results[key].append(valid_res[key])
                    except Exception as e:
                        if isinstance(e, gpytorch.utils.errors.NotPSDError):
                            message += '[Handled ERR] non-PSD cov\n'
                            raise e
                            #self.non_psd_cov = True # TODO: handle
                        else:
                            message += '[Unhandled ERR]'
                            raise e

                    # log info
                    for criterion in criteria:
                        # message += ' - Train-' + criterion + ': {:2.2f}'.format(train_results[criterion][-1])
                        message +=  ', Valid-' + criterion + ': {:2.4f}'.format(valid_results[criterion][-1])

                self.logger.info(message)

                # update the best posterior if early_stopping
                if early_stopping and itr>1:
                    for criterion in criteria:
                        if valid_results[criterion][-1] < min_criterion[criterion]:
                            min_criterion[criterion] = valid_results[criterion][-1]
                            self.best_posterior[criterion] = copy.deepcopy(self.posterior)

        self.fitted = True
        # set back to best posterior if early stopping
        if early_stopping and (not self.best_posterior[criteria[0]] is None):
            self.posterior = self.best_posterior[criteria[0]]

        return loss.item(), history



    def predict(self, context_x, context_y, test_x, n_posterior_samples=100, mode='Bayes', return_density=False):
        """
        computes the predictive distribution of the targets p(t|test_x, test_context_x, context_y)

        Args:
            context_x: (ndarray) context input data for which to compute the posterior
            context_y: (ndarray) context targets for which to compute the posterior
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            n_posterior_samples: (int) number of samples from posterior to average over
            mode: (std) either of ['Bayes' , 'MAP']
            return_density: (bool) whether to return result as mean and std ndarray or as MultivariateNormal pytorch object

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation corresponding to p(t|test_x, test_context_x, context_y)
        """

        assert mode in ['bayes', 'Bayes', 'MAP', 'map']

        context_x, context_y = _handle_input_dimensionality(context_x, context_y)
        test_x = _handle_input_dimensionality(test_x)
        assert test_x.shape[1] == context_x.shape[1]

        # normalize data and convert to tensor
        context_x, context_y = self._prepare_data_per_task(context_x, context_y)

        test_x = self._normalize_data(X=test_x, Y=None)
        test_x = torch.from_numpy(test_x).float().to(device)

        with torch.no_grad():

            if mode == 'Bayes' or mode == 'bayes':
                pred_dist = self.get_pred_dist(context_x, context_y, test_x, n_post_samples=n_posterior_samples)
                pred_dist = AffineTransformedDistribution(pred_dist, normalization_mean=self.y_mean,
                                                    normalization_std=self.y_std)
                pred_dist = EqualWeightedMixtureDist(pred_dist, batched=True)
            else:
                pred_dist = self.get_pred_dist_map(context_x, context_y, test_x)
                pred_dist = AffineTransformedDistribution(pred_dist, normalization_mean=self.y_mean,
                                                    normalization_std=self.y_std)


            if return_density:
                return pred_dist
            else:
                pred_mean = pred_dist.mean.cpu().numpy()
                pred_std = pred_dist.stddev.cpu().numpy()
                return pred_mean, pred_std



    def state_dict(self):
        state_dict = {
            'optimizer': self.optimizer.state_dict(),
            'model': self.task_dicts[0]['model'].state_dict()
        }
        for task_dict in self.task_dicts:
            for key, tensor in task_dict['model'].state_dict().items():
                assert torch.all(state_dict['model'][key] == tensor).item()
        return state_dict

    def load_state_dict(self, state_dict):
        for task_dict in self.task_dicts:
            task_dict['model'].load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def _setup_model_inference(
        self, mean_module_str, covar_module_str, mean_nn_layers, kernel_nn_layers,
        nonlinearity_hidden_m, nonlinearity_hidden_k, nonlinearity_output_m, nonlinearity_output_k,
        likelihood_str, cov_type):

        """ random gp model """
        self.hyper_post = HyperPosterior(
            input_dim=self.input_dim, feature_dim=self.feature_dim,
            prior_factor=self.prior_factor,
            hyper_prior_dict=self.hyper_prior_dict,
            covar_module_str=covar_module_str,
            mean_module_str=mean_module_str,
            mean_nn_layers=mean_nn_layers,
            kernel_nn_layers=kernel_nn_layers,
            nonlinearity_hidden_m=nonlinearity_hidden_m,
            nonlinearity_hidden_k=nonlinearity_hidden_k,
            nonlinearity_output_m=nonlinearity_output_m,
            nonlinearity_output_k=nonlinearity_output_k,
            likelihood_str=likelihood_str,
            optimize_noise=self.optimize_noise, noise_std=self.noise_std,
            optimize_lengthscale=self.optimize_lengthscale, lengthscale_fix=self.lengthscale_fix,
            logger=self.logger)


        param_shapes_dict = self.hyper_post.parameter_shapes()

        """ variational posterior """
        self.posterior = RandomGPPosterior(
            param_shapes_dict, cov_type=cov_type, param_dists=self.hyper_post._param_dists)


    """ define negative ELBO """
    def get_neg_elbo(self, tasks_dicts):
        # tile data to svi_batch_shape
        data_tuples_tiled = _tile_data_tuples(tasks_dicts, self.svi_batch_size)
        param_sample = self.posterior.rsample(sample_shape=(self.svi_batch_size,)) # TODO: does it ensure there is one sample from each variational  dist?
        elbo = self.hyper_post.log_prob(param_sample, data_tuples_tiled) - self.posterior.log_prob(param_sample)
        # NOTE: originally, the second term was wrong
        # self.prior_factor * self.posterior.log_prob(param_sample)
        assert elbo.ndim == 1 and elbo.shape[0] == self.svi_batch_size
        return - torch.mean(elbo)

    # self.get_neg_elbo = get_neg_elbo

    """ define predictive dist """
    def get_pred_dist(self, x_context, y_context, x_valid, n_post_samples=100):

        with torch.no_grad():
            x_context = x_context.view(torch.Size((1,)) + x_context.shape).repeat(n_post_samples, 1, 1)
            y_context = y_context.view(torch.Size((1,)) + y_context.shape).repeat(n_post_samples, 1)
            x_valid = x_valid.view(torch.Size((1,)) + x_valid.shape).repeat(n_post_samples, 1, 1)
            # print(x_context.shape, y_context.shape, x_valid.shape)

            # print('posterior mode ', self.posterior.mode.shape)
            param_sample = self.posterior.sample(sample_shape=(n_post_samples,))
            # print('param sample ', param_sample.shape)
            gp_fn = self.hyper_post.get_forward_fn(param_sample) # copy of self.hyper_post.gp with set param. self.gp is VectorizedGP
            gp, likelihood = gp_fn(x_context, y_context, train=False) # gp is LearnedGPRegressionModel, likelihood is Gaussian Likelihood
            # prediction for time-series data
            if self.ts_data:
                pred_mean = torch.zeros(
                    (self.num_particles, x_valid.shape[1])).to(device)
                pred_cov= torch.zeros(
                    (self.num_particles, x_valid.shape[1], x_valid.shape[1])).to(device) # diagonal matrix

                for point_num in np.arange(x_valid.shape[1]):
                    pred_dist_tmp = likelihood(gp(torch.reshape(
                        x_valid[:, point_num, :],
                        (x_valid.shape[0], 1, x_valid.shape[-1])
                    )))
                    pred_mean[:, point_num] = pred_dist_tmp.mean.flatten()
                    pred_cov[:, point_num, point_num] = pred_dist_tmp.covariance_matrix.flatten()

                pred_dist = gpytorch.distributions.MultivariateNormal(pred_mean, pred_cov)
            # prediction for non-time-series data (the whole test set at once)
            else:
                pred_dist = likelihood(gp(x_valid))
        return pred_dist

    def get_pred_dist_map(self, x_context, y_context, x_valid):
        # beta can be provided in kwargs to ssimulate generalized marginal likelihood
        # if not given, beta=num_samples and the likelihood is returned

        with torch.no_grad():
            x_context = x_context.view(torch.Size((1,)) + x_context.shape).repeat(1, 1, 1)
            y_context = y_context.view(torch.Size((1,)) + y_context.shape).repeat(1, 1)
            x_valid = x_valid.view(torch.Size((1,)) + x_valid.shape).repeat(1, 1, 1)
            param = self.posterior.mode
            param = param.view(torch.Size((1,)) + param.shape).repeat(1, 1)

            gp_fn = self.hyper_post.get_forward_fn(param)
            gp, likelihood = gp_fn.forward(x_context, y_context, train=False)
            pred_dist = likelihood(gp(x_valid))
        return MultivariateNormal(pred_dist.loc, pred_dist.covariance_matrix[0])

        # TODO: if beta neq m, the output will not be Gaussian anymore. don't know how to estimate it.

    # self.get_pred_dist = get_pred_dist
    # self.get_pred_dist_map = get_pred_dist_map

    def _setup_optimizer(self, optimizer, lr, lr_decay):
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.posterior.parameters(), lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.posterior.parameters(), lr=lr)
        else:
            raise NotImplementedError('Optimizer must be Adam or SGD')

        if lr_decay < 1.0:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1000, gamma=lr_decay)
        else:
            self.lr_scheduler = DummyLRScheduler()

    def _vectorize_pred_dist(self, pred_dist):
        multiv_normal_batched = pred_dist.dists
        normal_batched = torch.distributions.Normal(multiv_normal_batched.mean, multiv_normal_batched.stddev)
        return EqualWeightedMixtureDist(normal_batched, batched=True, num_dists=multiv_normal_batched.batch_shape[0])


def _tile_data_tuples(tasks_dicts, tile_size):
    train_data_tuples_tiled = []
    for task_dict in tasks_dicts:
        x_data, y_data = task_dict['train_x'], task_dict['train_y']
        x_data = x_data.view(torch.Size((1,)) + x_data.shape).repeat(tile_size, 1, 1)
        y_data = y_data.view(torch.Size((1,)) + y_data.shape).repeat(tile_size, 1)
        # print('x data shape in tile', x_data.shape)
        train_data_tuples_tiled.append((x_data, y_data))
    return train_data_tuples_tiled


if __name__ == "__main__":
    """ 1) Generate some training data from GP prior """
    from experiments.data_sim import GPFunctionsDataset

    data_sim = GPFunctionsDataset(random_state=np.random.RandomState(26))

    meta_train_data = data_sim.generate_meta_train_data(n_tasks=10, n_samples=40)
    meta_test_data = data_sim.generate_meta_test_data(n_tasks=10, n_samples_context=40, n_samples_test=160)

    NN_LAYERS = (32, 32)

    plot = False
    from matplotlib import pyplot as plt

    if plot:
        for x_train, y_train in meta_train_data:
            plt.scatter(x_train, y_train)
        plt.title('sample from the GP prior')
        plt.show()

    """ 2) Classical mean learning based on mll """

    print('\n ---- GPR VI meta-learning ---- ')

    torch.set_num_threads(2)

    for prior_factor in [1e-3 / 40.]:
        gp_model = GPRegressionMetaLearnedVI(meta_train_data, num_iter_fit=2000, prior_factor=prior_factor, svi_batch_size=10, task_batch_size=2,
                                             covar_module_str='SE', mean_module_str='NN', mean_nn_layers=NN_LAYERS, kernel_nn_layers=NN_LAYERS, cov_type='diag')


        for i in range(10):
            itrs = 0
            gp_model.meta_fit(valid_tuples=meta_test_data, log_period=500, max_iter_fit=2000)
            itrs += 2000

            x_test = np.linspace(-5, 5, num=150)
            x_context, t_context, _, _ = meta_test_data[0]
            pred_mean, pred_std = gp_model.predict(x_context, t_context, x_test)
            ucb, lcb = gp_model.confidence_intervals(x_context, t_context, x_test, confidence=0.9)

            plt.scatter(x_context, t_context)
            plt.plot(x_test, pred_mean)
            plt.fill_between(x_test, lcb, ucb, alpha=0.2)
            plt.title('GPR meta VI (prior-factor =  %.4f) itrs = %i' % (prior_factor, itrs))
            plt.show()