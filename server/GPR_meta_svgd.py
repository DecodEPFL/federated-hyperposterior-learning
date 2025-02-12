import torch, gpytorch, time, warnings
import numpy as np
from gpytorch.utils.warnings import GPInputWarning

from server.models import AffineTransformedDistribution, EqualWeightedMixtureDist
from server.hyper_posterior import HyperPosterior, IllCondException
from server.util import _handle_input_dimensionality, DummyLRScheduler, WrapLogger, warning_show, warning_format, print_gpvec_prior_params
from server.svgd import SVGD, RBF_Kernel, IMQSteinKernel
from server.abstract import RegressionModelMetaLearned
from config import device

warnings.formatwarning = warning_format
#warnings.showwarning = warning_show

class GPRegressionMetaLearnedSVGD(RegressionModelMetaLearned):

    def __init__(self, meta_train_data, num_iter_fit, feature_dim,
                 covar_module_str, mean_module_str,likelihood_str,
                 prior_factor, hyper_prior_dict={},
                 mean_nn_layers=(32, 32), kernel_nn_layers=(32, 32),
                 nonlinearity_hidden_m=torch.tanh, nonlinearity_hidden_k=torch.tanh,
                 nonlinearity_output_m=None, nonlinearity_output_k=None,
                 optimizer='Adam', lr=1e-3, lr_decay=1.0, kernel='RBF', bandwidth=None, num_particles=10,
                 task_batch_size=-1, normalize_data=True, random_seed=None,
                 optimize_noise=True, noise_std=None,
                 optimize_lengthscale=True, lengthscale_fix=None,
                 logger=None, initial_particles=None, ts_data=False):

        """
        PACOH-SVGD: Stein Variational Gradient Descent on PAC-optimal hyper-posterior.
        Meta-learns a set of GP priors (i.e. mean and kernel function)

        Args:
            meta_train_data: list of tuples of ndarrays[(train_x_1, train_t_1), ..., (train_x_n, train_t_n)]
            num_iter_fit: (int) number of gradient steps for fitting the parameters
            feature_dim: (int) output dimensionality of NN feature map for kernel function
            prior_factor: (float) weighting of the hyper-prior (--> meta-regularization parameter)
            hyper_prior_dict (float): dictionary pf std of Gaussian hyper-prior on weights, biases, length-scale, noise
            covar_module_str: (gpytorch.mean.Kernel) optional kernel module, default: RBF kernel
            mean_module_str: (gpytorch.mean.Mean) optional mean module, default: ZeroMean
            mean_nn_layers: (tuple) hidden layer sizes of mean NN
            kernel_nn_layers: (tuple) hidden layer sizes of kernel NN
            optimizer: (str) type of optimizer to use - must be either 'Adam' or 'SGD'
            lr: (float) learning rate for prior parameters
            lr_decay: (float) lr rate decay multiplier applied after every 1000 steps
            kernel (std): SVGD kernel, either 'RBF' or 'IMQ'
            bandwidth (float): bandwidth of kernel, if None the bandwidth is chosen via heuristic
            num_particles: (int) number particles to approximate the hyper-posterior
            task_batch_size: (int) mini-batch size of tasks for estimating gradients
            normalize_data: (bool) whether the data should be normalized
            ts_data: specify if the data is time-series. If True, test points are passed to the model one by one.
            random_seed: (int) seed for pytorch
        """
        super().__init__(normalize_data, random_seed)

        assert mean_module_str in ['NN', 'constant', 'zero', 'linear', 'linear_no_bias']
        assert covar_module_str in ['NN', 'SNN', 'SE', 'linear', 'periodic', 'zero'] #or isinstance(covar_module_str, gpytorch.kernels.Kernel)
        assert likelihood_str in ['Gaussian', 'FixedNoise']
        assert optimizer in ['Adam', 'SGD']
        self.num_iter_fit, self.prior_factor, self.feature_dim = num_iter_fit, prior_factor, feature_dim
        self.optimize_noise, self.noise_std = optimize_noise, noise_std
        self.optimize_lengthscale, self.lengthscale_fix = optimize_lengthscale, lengthscale_fix
        self.hyper_prior_dict = hyper_prior_dict
        self.num_particles = num_particles
        self.best_particles = None
        self.ts_data = ts_data

        if task_batch_size < 1:
            self.task_batch_size = len(meta_train_data)
        else:
            self.task_batch_size = min(task_batch_size, len(meta_train_data))

        # Check that data all has the same size
        self._check_meta_data_shapes(meta_train_data)
        self._compute_normalization_stats(meta_train_data)


        """ --- Setup model & inference --- """
        self._setup_model_inference(
            mean_module_str=mean_module_str, covar_module_str=covar_module_str,
            mean_nn_layers=mean_nn_layers, kernel_nn_layers=kernel_nn_layers,
            nonlinearity_hidden_m=nonlinearity_hidden_m, nonlinearity_hidden_k=nonlinearity_hidden_k,
            nonlinearity_output_m=nonlinearity_output_m, nonlinearity_output_k=nonlinearity_output_k,
            likelihood_str=likelihood_str,
            kernel=kernel, bandwidth=bandwidth,
            optimizer=optimizer, lr=lr, lr_decay=lr_decay,
            initial_particles=initial_particles)

        """ Disable Privacy"""
        self.setup_privacy()

        # Setup components that are different across tasks
        self.task_dicts = []
        for train_x, train_y in meta_train_data:
            task_dict = {}
            x_tensor, y_tensor = self._prepare_data_per_task(train_x, train_y) # are casted to device
            task_dict['train_x'], task_dict['train_y'] = x_tensor, y_tensor
            self.task_dicts.append(task_dict)
        self.fitted = False
        self.over_fitted = False
        self.non_psd_cov, self.unknown_err = False, False
        self.logger=WrapLogger(logger) if not isinstance(logger, WrapLogger) else logger




    def setup_privacy(self, clip_norm=None, private=False, epsilon=None, delta=None, record_norms=False):
        self.svgd.setup_privacy(clip_norm=clip_norm, private=private,
                                epsilon=epsilon, delta=delta, record_norms=record_norms)


    def meta_fit(
        self, criteria='rmse', over_fit_margin=None, cont_fit_margin=None, max_iter_fit=None,
        early_stopping=True, valid_tuples=None, log_period=500, record_params=False):
        """
        fits the hyper-posterior particles with SVGD

        Args:
            over_fit_margin: abrupt training if slope of valid RMSE over one log_period > over_fit_margin (set to None to disable early stopping)
            cont_fit_margin: continue training for more iters if slope of valid RMSE over one log_period < - cont_fit_margin
            max_iter_fit: max iters to extend training
            early_stopping: return model at an evaluated iteration with the lowest valid RMSE
            valid_tuples: list of valid tuples, i.e. [(test_context_x_1, test_context_t_1, test_x_1, test_t_1), ...]
            log_period (int) number of steps after which to print stats
            criteria: return one model minimizing each criterion in the list criteria.
                      if more than one criterion is provided, the first one is used for diagnosing over-fitting or extending training
        """
        if not isinstance(criteria, list):
            criteria = [criteria]
        criteria = [criterion.lower() for criterion in criteria]
        for criterion in criteria:
            assert criterion in ['rmse', 'rsmse', 'calibr', 'nll']
        assert (valid_tuples is None) or (all([len(valid_tuple) == 4 for valid_tuple in valid_tuples]))
        if not valid_tuples is None:
            train_tuples = [(x[0], x[1], x[0], x[1]) for x in valid_tuples]
        #num_params = list(self.optimizer.param_groups[0]['params'][0].shape)[0]
        if record_params and self.num_iter_fit*self.particles.shape[0]*self.particles.shape[1]>5e4:
            self.logger.info('[WARNING] output log would be larger than 500 MB. Disabled param recording')
            record_params = False

        if early_stopping:
            self.best_particles = dict.fromkeys(criteria, None) # track the best particles not the whole history
            min_criterion = dict.fromkeys(criteria, 1e6)

        # initialize losses
        valid_results = dict.fromkeys(criteria)
        train_results = dict.fromkeys(criteria)
        params = [None]
        if valid_tuples is not None:
            # initial evaluation on validation data
            valid_res = self.eval_datasets(valid_tuples)
            for key in valid_results.keys():
                valid_results[key] = [valid_res[key]]
            # initial evaluation on validation data
            try:
                train_res = self.eval_datasets(train_tuples)
                for key in train_results.keys():
                    train_results[key] = [train_res[key]]
            except Exception as e:
                if isinstance(e, gpytorch.utils.warnings.GPInputWarning):
                    pass
                else:
                    self.logger.info('[Unhandled ERR]' + type(e).__name__ + '\n')
                    self.logger.info(e)
                    self.unknown_err = True

        last_params = self.particles.detach().clone() # params in the last iteration
        if record_params:
            params[0]= last_params.detach().numpy()

        t = time.time()
        itr = 1
        while itr <= self.num_iter_fit:
            task_dict_batch = self.rds_numpy.choice(self.task_dicts, size=self.task_batch_size)
            # --- take a step ---
            try:
                self.svgd_step(task_dict_batch)
            except Exception as e:
                if isinstance(e, gpytorch.utils.errors.NotPSDError):
                    self.logger.info('[Handled ERR] non-PSD cov\n')
                    self.non_psd_cov = True
                elif isinstance(e, IllCondException):
                    self.logger.info('[Handled ERR] ill-conditioned cov\n') #TODO: check
                    self.non_psd_cov = True # TODO: change flag name
                else:
                    self.logger.info('[Unhandled ERR]' + type(e).__name__ + '\n')
                    self.logger.info(e)
                    self.unknown_err = True

            # --- print stats ---
            if (itr == 1 or itr % log_period == 0) and (not self.non_psd_cov or self.unknown_err):
                duration = time.time() - t
                t = time.time()
                message = 'Iter %d/%d - Time %.2f sec' % (itr, self.num_iter_fit, duration)

                # print info about norm of gradients
                if itr >= 100:
                    message += ' -  av. grad norm: %.3f' % \
                               (sum(self.svgd.noisy_norm[-log_period:]).cpu().numpy()/100)
                else:
                    message += ' -  av. grad norm: %.3f' % (self.svgd.noisy_norm[-1])

                # if validation data is provided  -> compute the valid log-likelihood
                if valid_tuples is not None:
                    # evaluate on validation set
                    try:
                        valid_res = self.eval_datasets(valid_tuples)
                        for key in valid_results.keys():
                            valid_results[key].append(valid_res[key])
                    except Exception as e:
                        if isinstance(e,gpytorch.utils.errors.NotPSDError):
                            message += '[Handled ERR] non-PSD cov\n'
                            self.non_psd_cov = True
                        else:
                            message += '[Unhandled ERR]'
                            self.logger.info(e)
                            self.unknown_err = True
                    # evaluate on train set
                    try:
                        train_res = self.eval_datasets(train_tuples)
                        for key in train_results.keys():
                            train_results[key].append(train_res[key])
                    except Exception as e:
                        if isinstance(e, gpytorch.utils.warnings.GPInputWarning):
                            pass
                        else:
                            self.logger.info('[Unhandled ERR]' + type(e).__name__ + '\n')
                            self.logger.info(e)
                            self.unknown_err = True
                    # log info
                    for criterion in criteria:
                        message += ' - Train-' + criterion + ': {:2.2f}'.format(train_results[criterion][-1])
                        message +=  ', Valid-' + criterion + ': {:2.2f}'.format(valid_results[criterion][-1])

                    # check over-fitting
                    if not over_fit_margin is None:
                        if valid_results[criteria[0]][-1]-valid_results[criteria[0]][-2] >= over_fit_margin * log_period:
                            self.over_fitted = True
                            message += '\n[WARNING] model over-fitted according to ' + criteria[0]

                    # check continue training
                    if (not ((cont_fit_margin is None) or (max_iter_fit is None))) and (itr+log_period <= max_iter_fit) and (itr+log_period > self.num_iter_fit):
                        if valid_results[criteria[0]][-1]-valid_results[criteria[0]][-2] <= - abs(cont_fit_margin) * log_period:
                            self.num_iter_fit += log_period
                            message += '\n[Info] extended training according to ' + criteria[0]

                    # update the best particles if early_stopping
                    if early_stopping and itr>1:
                        for criterion in criteria:
                            if valid_results[criterion][-1] < min_criterion[criterion]:
                                min_criterion[criterion] = valid_results[criterion][-1]
                                self.best_particles[criterion] = self.particles.detach().clone()


                # log info
                self.logger.info(message)

            # go one iter back if non-psd
            if self.non_psd_cov or self.unknown_err:
                self.particles = last_params.detach().clone() # set back params to the previous iteration

            # stop training
            if self.over_fitted or self.non_psd_cov or self.unknown_err:
                break

            # update learning rate
            self.lr_scheduler.step()
            # go to next iter
            last_params = self.particles.detach().clone() # num_rows = num_particles, columns are prior params. for variance, must apply softplus
            if record_params:
                params.append(last_params.cpu().detach().numpy())
            itr = itr+1


        self.fitted = True if not (self.non_psd_cov or self.unknown_err) else False
        self.history={'original_norm': self.svgd.original_norm,
                      'noisy_norm': self.svgd.noisy_norm, 'clipped_norm': self.svgd.clipped_norm,
                      'valid_results':valid_results, 'train_results':train_results,
                      'fitted':self.fitted, 'over_fitted': self.over_fitted,
                      'non_psd_cov': self.non_psd_cov, 'unknown_err':self.unknown_err}
        if record_params:
            self.history['params'] = params

        # set back to best particles if early stopping
        if early_stopping and (not self.best_particles[criteria[0]] is None):
            self.particles = self.best_particles[criteria[0]]



    def predict(self, context_x, context_y, test_x, return_density=False, use_weights=True):
        """
        Performs posterior inference (target training) with (context_x, context_y) as training data and then
        computes the predictive distribution of the targets p(y|test_x, test_context_x, context_y) in the test points
        Inputs are not normalized, output is not normalized
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
        context_x, context_y = self._prepare_data_per_task(context_x, context_y) # are casted to device

        test_x = self._normalize_data(X=test_x, Y=None)
        test_x = torch.from_numpy(test_x).float().to(device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
            # compute particle weights
            if use_weights:
                comb_weights = self.hyper_post.comb_weights(
                    self.particles, context_x, context_y, weights_hyp_prior=None)
            else:
                comb_weights = None
            # self.logger.info(comb_weights)

            '''
            pred_dist is MultiVariateNormal with lazy mean, covariance_matrix
            mean is of torch.Size([num_particles, num_valid_samples])
            covariance_matrix is of torch.Size([num_particles, num_valid_samples, num_valid_samples])
            '''
            pred_dist = self.get_pred_dist(
                context_x, context_y, test_x)

            '''
            unnormalize model predictions and extract diagonal values of the covar and take sqrt
            mean is of torch.Size([num_particles, num_valid_samples])
            stddev is of torch.Size([num_particles, num_valid_samples])
            '''
            pred_dist = AffineTransformedDistribution(
                pred_dist, normalization_mean=self.y_mean,
                normalization_std=self.y_std)

            '''
            combine distributions resulting from different particles
            mean is of torch.Size([num_valid_samples])
            stddev is of torch.Size([num_valid_samples])
            '''
            pred_dist = EqualWeightedMixtureDist(
                pred_dist, batched=True, comb_weights=comb_weights)

            if return_density:
                return pred_dist
            else:
                pred_mean = pred_dist.mean.cpu().numpy()
                pred_std = pred_dist.stddev.cpu().numpy()
                return pred_mean, pred_std



    def _setup_model_inference(
        self, mean_module_str, covar_module_str, mean_nn_layers, kernel_nn_layers,
        nonlinearity_hidden_m, nonlinearity_hidden_k, nonlinearity_output_m, nonlinearity_output_k,
        likelihood_str, kernel, bandwidth, optimizer, lr, lr_decay, initial_particles):

        """define a genertic hyper-posterior"""
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


        """initialize particles"""
        # sample initial particle locations from hyper-prior
        initial_sampled_particles = self.hyper_post.sample_params_from_prior(shape=(self.num_particles,))
        # set given initial particles
        if not initial_particles is None:
            idx = 0
            for name, shape in self.hyper_post.gp.parameter_shapes().items():
                idx_next = idx + shape[-1]
                if name in initial_particles.keys():
                    self.logger.info('initialized ' + name)
                    initial_sampled_particles[:, idx:idx_next] = torch.tensor(initial_particles[name]).float().to(device)
                idx = idx_next
        self.initial_particles = initial_sampled_particles
        self.particles = self.initial_particles.detach().clone()        # set particles to the initial value


        self._setup_optimizer(optimizer, lr, lr_decay)

        """ Setup SVGD inference"""
        if kernel == 'RBF':
            kernel = RBF_Kernel(bandwidth=bandwidth)
        elif kernel == 'IMQ':
            kernel = IMQSteinKernel(bandwidth=bandwidth)
        else:
            raise NotImplementedError
        self.svgd = SVGD(self.hyper_post, kernel, optimizer=self.optimizer)

        """ define svgd step """

        def svgd_step(tasks_dicts):
            # tile data to svi_batch_shape
            train_data_tuples_tiled = []
            for task_dict in tasks_dicts:
                x_data, y_data = task_dict['train_x'], task_dict['train_y']
                x_data = x_data.view(torch.Size((1,)) + x_data.shape).repeat(self.num_particles, 1, 1)
                y_data = y_data.view(torch.Size((1,)) + y_data.shape).repeat(self.num_particles, 1)
                train_data_tuples_tiled.append((x_data, y_data))

            self.svgd.step(self.particles, train_data_tuples_tiled)

        """ define predictive dist """

        def get_pred_dist(x_context, y_context, x_valid):
            with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
                x_context = x_context.view(torch.Size((1,)) + x_context.shape).repeat(self.num_particles, 1, 1)
                y_context = y_context.view(torch.Size((1,)) + y_context.shape).repeat(self.num_particles, 1)
                x_valid = x_valid.view(torch.Size((1,)) + x_valid.shape).repeat(self.num_particles, 1, 1)
                gp_fn = self.hyper_post.get_forward_fn(self.particles)
                #if self.covar_module_str == 'zero' # TODO: normal(mean(x_test), sigma_n I)
                gp, likelihood = gp_fn.forward(x_context, y_context, train=False) # gp is LearnedGPRegressionModel with training=False
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
                # NOTE: if train points are correlated, with FixedNoise likelihood gp(x_valid).mean is Nan
            return pred_dist

        self.svgd_step = svgd_step
        self.get_pred_dist = get_pred_dist

    def _setup_optimizer(self, optimizer, lr, lr_decay):
        assert hasattr(self, 'particles'), "SVGD must be initialized before setting up optimizer"

        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam([self.particles], lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD([self.particles], lr=lr)
        else:
            raise NotImplementedError('Optimizer must be Adam or SGD')

        if lr_decay < 1.0:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1000, gamma=lr_decay)
        else:
            self.lr_scheduler = DummyLRScheduler()

    def _vectorize_pred_dist(self, pred_dist):
        multiv_normal_batched = pred_dist.dists
        num_dists=multiv_normal_batched.batch_shape[0]
        normal_batched = torch.distributions.Normal(multiv_normal_batched.mean, multiv_normal_batched.stddev)
        return EqualWeightedMixtureDist(
            normal_batched, batched=True,
            num_dists=num_dists, comb_weights=pred_dist.comb_weights)


    def print_gp_prior_params(self, print_nn_weights=False):
        gpvec = self.hyper_post.get_forward_fn(self.particles)
        return print_gpvec_prior_params(gpvec, print_nn_weights=print_nn_weights)



if __name__ == "__main__":
    """ 1) Generate some training data from GP prior """
    from experiments.data_sim import GPFunctionsDataset

    data_sim = GPFunctionsDataset(random_state=np.random.RandomState(26))

    # meta_train_data = data_sim.generate_meta_train_data(n_tasks=5, n_samples=40)
    meta_test_data = data_sim.generate_meta_test_data(n_tasks=5, n_samples_context=40, n_samples_test=160)
    meta_train_data = [(context_x, context_y) for context_x, context_y, _, _ in meta_test_data]

    NN_LAYERS = (16, 16)
    n_iter = 1

    plot = True
    from matplotlib import pyplot as plt

    if plot:
        for x_train, y_train in meta_train_data:
            plt.scatter(x_train, y_train)
        plt.title('sample from the GP prior')
        plt.show()

    for prior_factor in [1e-3]:
        gp_model = GPRegressionMetaLearnedSVGD(meta_train_data, num_iter_fit=n_iter, prior_factor=prior_factor, num_particles=1,
                                             covar_module_str='linear', mean_module_str='constant', mean_nn_layers=NN_LAYERS, kernel_nn_layers=NN_LAYERS,
                                             bandwidth=0.5, task_batch_size=2)

        iterations = 0
        for i in range(10):
            gp_model.meta_fit(valid_tuples=meta_test_data, log_period=500, n_iter=n_iter)
            iterations += n_iter

            x_test = np.linspace(-5, 5, num=150)
            x_context, t_context, _, _ = meta_test_data[0]
            pred_mean, pred_std = gp_model.predict(x_context, t_context, x_test)
            ucb, lcb = gp_model.confidence_intervals(x_context, t_context, x_test, confidence=0.9)

            plt.scatter(x_context, t_context)
            plt.plot(x_test, pred_mean)
            plt.fill_between(x_test, lcb, ucb, alpha=0.2)
            plt.title('GPR meta SVGD (prior-factor =  %.4f) iterations = %i'%(prior_factor, iterations))
            plt.show()