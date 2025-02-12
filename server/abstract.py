import numpy as np
import torch, warnings
from server.util import get_logger, _handle_input_dimensionality, warning_show, warning_format
from config import device

#warnings.showwarning = warning_show
warnings.formatwarning = warning_format

class RegressionModel:

    def __init__(self, normalize_data=True, random_seed=None):
        self.normalize_data = normalize_data
        self.logger = get_logger()
        self.input_dim = None
        self.output_dim = None
        self.n_train_samples = None
        self.x_train = None
        self.y_train = None

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed+1)

    def predict(self, x_valid, return_density=False, **kwargs):
        raise NotImplementedError

    def eval(self, x_valid, y_valid, **kwargs):
        """
        Computes the average test log likelihood and the rmse on test data

        Args:
            x_valid: (ndarray) test input data of shape (n_samples, ndim_x)
            y_valid: (ndarray) test target data of shape (n_samples, 1)

        Returns: (neg avg_log_likelihood, rmse)

        """
        # convert to tensors
        x_valid, y_valid = _handle_input_dimensionality(x_valid, y_valid)
        y_valid_tensor = torch.from_numpy(y_valid).contiguous().float().flatten().to(device)

        with torch.no_grad():
            pred_dist = self.predict(x_valid, return_density=True, *kwargs)
            avg_log_likelihood = pred_dist.log_prob(y_valid_tensor) / y_valid_tensor.shape[0]
            rmse = torch.mean(torch.pow(pred_dist.mean.reshape(y_valid_tensor.shape)-y_valid_tensor, 2)).sqrt()

            pred_dist_vect = self._vectorize_pred_dist(pred_dist)
            calibr_error = self._calib_error(pred_dist_vect, y_valid_tensor)

            return -1*avg_log_likelihood.cpu().item(), rmse.cpu().item(), calibr_error.cpu().item()

    def confidence_intervals(self, x_valid, confidence=0.9, **kwargs):
        pred_dist = self.predict(x_valid, return_density=True, **kwargs)
        pred_dist = self._vectorize_pred_dist(pred_dist)

        alpha = (1 - confidence) / 2
        ucb = pred_dist.icdf(torch.ones(x_valid.size) * (1 - alpha))
        lcb = pred_dist.icdf(torch.ones(x_valid.size) * alpha)
        return ucb, lcb

    def _calib_error(self, pred_dist_vectorized, y_valid_tensor):
        return _calib_error(pred_dist_vectorized, y_valid_tensor)

    def _compute_normalization_stats(self, X, Y):
        # save mean and variance of data for normalization
        if self.normalize_data:
            self.x_mean, self.y_mean = np.mean(X, axis=0), np.mean(Y, axis=0)
            self.x_std, self.y_std = np.std(X, axis=0) + 1e-8, np.std(Y, axis=0) + 1e-8
        else:
            self.x_mean, self.y_mean = np.zeros(X.shape[1]), np.zeros(Y.shape[1])
            self.x_std, self.y_std = np.ones(X.shape[1]), np.ones(Y.shape[1])

    def _normalize_data(self, X, Y=None):
        assert hasattr(self, "x_mean") and hasattr(self, "x_std"), "requires computing normalization stats beforehand"
        assert hasattr(self, "y_mean") and hasattr(self, "y_std"), "requires computing normalization stats beforehand"

        X_normalized = (X - self.x_mean[None, :]) / self.x_std[None, :]

        if Y is None:
            return X_normalized
        else:
            Y_normalized = (Y - self.y_mean[None, :]) / self.y_std[None, :]
            return X_normalized, Y_normalized


    def _unnormalize_pred(self, pred_mean, pred_std):
        assert hasattr(self, "x_mean") and hasattr(self, "x_std"), "requires computing normalization stats beforehand"
        assert hasattr(self, "y_mean") and hasattr(self, "y_std"), "requires computing normalization stats beforehand"

        if self.normalize_data:
            assert pred_mean.ndim == pred_std.ndim == 2 and pred_mean.shape[1] == pred_std.shape[1] == self.output_dim
            if isinstance(pred_mean, torch.Tensor) and isinstance(pred_std, torch.Tensor):
                y_mean_tensor, y_std_tensor = torch.tensor(self.y_mean).float().to(device), torch.tensor(self.y_std).float().to(device)
                pred_mean = pred_mean.mul(y_std_tensor[None, :]) + y_mean_tensor[None, :]
                pred_std = pred_std.mul(y_std_tensor[None, :])
            else:
                pred_mean = pred_mean.multiply(self.y_std[None, :]) + self.y_mean[None, :]
                pred_std = pred_std.multiply(self.y_std[None, :])

        return pred_mean, pred_std

    def _initial_data_handling(self, x_train, y_train):
        x_train, y_train = _handle_input_dimensionality(x_train, y_train)
        self.input_dim, self.output_dim = x_train.shape[-1], y_train.shape[-1]
        self.n_train_samples = x_train.shape[0]

        # b) normalize data to exhibit zero mean and variance
        self._compute_normalization_stats(x_train, y_train)
        x_train_normalized, y_train_normalized = self._normalize_data(x_train, y_train)

        # c) Convert the data into pytorch tensors
        self.x_train = torch.from_numpy(x_train_normalized).contiguous().float().to(device)
        self.y_train = torch.from_numpy(y_train_normalized).contiguous().float().to(device)

        return self.x_train, self.y_train

    def _vectorize_pred_dist(self, pred_dist):
        raise NotImplementedError

class RegressionModelMetaLearned:

    def __init__(self, normalize_data, random_seed=None):
        self.normalize_data = normalize_data
        self.logger = get_logger()
        self.input_dim = None
        self.output_dim = None

        if random_seed is not None:
            torch.manual_seed(random_seed)
            self.rds_numpy = np.random.RandomState(random_seed + 1)
        else:
            self.rds_numpy = np.random

    def predict(self, x_train, y_train, x_valid, **kwargs):
        raise NotImplementedError

    def eval(self, x_train, y_train, x_valid, y_valid, flatten_y=True, **kwargs):
        """
        Performs posterior inference (target training) with (x_train, y_train) as training data and then
        computes the average test log likelihood, rmse and calibration error on (x_valid, y_valid)

        Args:
            x_train: (ndarray) train input data for which to compute the posterior
            y_train: (ndarray) train targets for which to compute the posterior
            x_valid: (ndarray) valid input data of shape (n_samples, ndim_x)
            y_valid: (ndarray) valid targets data of shape (n_samples, 1)

        Returns: (neg avg_log_likelihood, rmse, calibr_error)

        """

        x_train, y_train = _handle_input_dimensionality(x_train, y_train)
        x_valid, y_valid = _handle_input_dimensionality(x_valid, y_valid)
        if flatten_y:
            y_valid_tensor = torch.from_numpy(y_valid).float().flatten().to(device)
        else:
            y_valid_tensor = torch.unsqueeze(torch.from_numpy(y_valid).float().to(device), dim=0)

        pred_dist = self.predict(x_train, y_train, x_valid, return_density=True, **kwargs)
        """ pred_dist is class 'server.models.EqualWeightedMixtureDist'
            pred_dist.mean and pred_dist.variance \in R^{num_test_samples}
            note: variance is a vector, b.c. pred_dist is stack of Gaussians at test points
            GP predictive dist has a full cov matrix, but here we only need the diagonal values.
            pred_dist.dists is AffineTransformedDistribution(), affine mix of posteriors corresponding to different particles """

        avg_log_likelihood = torch.mean(pred_dist.log_prob(y_valid_tensor) / y_valid_tensor.shape[0])
        rmse = torch.mean(torch.pow(pred_dist.mean - y_valid_tensor, 2)).sqrt()

        pred_dist_vect = self._vectorize_pred_dist(pred_dist)
        calibr_error = self._calib_error(pred_dist_vect, y_valid_tensor)

        return -1*avg_log_likelihood.cpu().item(), rmse.cpu().item(), calibr_error.cpu().item()

    def eval_datasets(self, clients_data, flatten_y=True, get_full_list=False, **kwargs):
        """
        Performs meta-testing on multiple tasks / datasets.
        Computes the average test log likelihood, the rmse and the calibration error over multiple test datasets

        Args:
            clients_data: list of test set tuples, i.e. [(test_x_train_1, y_valid_train_1, x_valid_1, y_valid_1), ...]
            get_std: return sample std of error measures
        Returns: (negative avg_log_likelihood, rmse, calibr_error) or with sample std
        """
        assert (all([len(valid_tuple) == 4 for valid_tuple in clients_data]))
        nll_list, rmse_list, calibr_err_list = list(zip(*[self.eval(*valid_data_tuple, flatten_y=flatten_y, **kwargs) for valid_data_tuple in clients_data]))
        # calculate rsmse
        rsmse_list = np.zeros(len(rmse_list))
        for client_num in np.arange(len(rmse_list)):
            rsmse_list[client_num] = rmse_list[client_num]/np.std(clients_data[client_num][3])
        if not get_full_list:
            return {
                'nll': np.mean(nll_list), 'calibr':np.mean(calibr_err_list),
                'rmse':np.mean(rmse_list), 'rsmse':np.mean(rsmse_list)}
        else:
            return {
                'nll': nll_list, 'calibr':calibr_err_list,
                'rmse':rmse_list, 'rsmse':rsmse_list}


    def confidence_intervals(self, x_train, y_train, x_valid, confidence=0.9, **kwargs):
        """
        Performs posterior inference (target training) with (x_train, y_train) as training data and then
        computes the confidence intervals corresponding to predictions p(y|x_valid, test_x_train, y_train) in the
        test points

        Args:
            x_train: (ndarray) train input data for which to compute the posterior
            y_train: (ndarray) train targets for which to compute the posterior
            x_valid: (ndarray) query input data of shape (n_samples, ndim_x)
            confidence: (float) confidence corresponding to the prediction interval, must be in [0,1)

        Returns:
            (ucb, lcb) upper and lower confidence bound
        """
        pred_dist = self.predict(x_train, y_train, x_valid, return_density=True, **kwargs)
        pred_dist = self._vectorize_pred_dist(pred_dist)

        alpha = (1-confidence) / 2
        ucb = pred_dist.icdf(torch.ones(x_valid.shape) * (1-alpha))
        lcb = pred_dist.icdf(torch.ones(x_valid.shape) * alpha)
        return ucb, lcb

    def _calib_error(self, pred_dist_vectorized, y_valid_tensor):
        return _calib_error(pred_dist_vectorized, y_valid_tensor)

    def _vectorize_pred_dist(self, pred_dist):
        raise NotImplementedError

    def _compute_normalization_stats(self, meta_y_trainuples):
        X_stack, Y_stack = list(zip(*[_handle_input_dimensionality(x_train, y_train) for x_train, y_train in meta_y_trainuples]))
        X, Y = np.concatenate(X_stack, axis=0), np.concatenate(Y_stack, axis=0)

        if self.normalize_data:
            self.x_mean, self.y_mean = np.mean(X, axis=0), np.mean(Y, axis=0)
            self.x_std, self.y_std = np.std(X, axis=0) + 1e-8, np.std(Y, axis=0) + 1e-8
        else:
            self.x_mean, self.y_mean = np.zeros(X.shape[1]), np.zeros(Y.shape[1])
            self.x_std, self.y_std = np.ones(X.shape[1]), np.ones(Y.shape[1])

    def _normalize_data(self, X, Y=None):
        assert hasattr(self, "x_mean") and hasattr(self, "x_std"), "requires computing normalization stats beforehand"
        assert hasattr(self, "y_mean") and hasattr(self, "y_std"), "requires computing normalization stats beforehand"

        X_normalized = (X - self.x_mean[None, :]) / self.x_std[None, :]

        if Y is None:
            return X_normalized
        else:
            Y_normalized = (Y - self.y_mean[None, :]) / self.y_std[None, :]
            return X_normalized, Y_normalized

    def _check_meta_data_shapes(self, meta_train_data):
        for i in range(len(meta_train_data)):
            meta_train_data[i] = _handle_input_dimensionality(*meta_train_data[i])
        self.input_dim = meta_train_data[0][0].shape[-1]
        self.output_dim = meta_train_data[0][1].shape[-1]

        assert all([self.input_dim == x_train.shape[-1] and self.output_dim == y_train.shape[-1] for x_train, y_train in meta_train_data])

    def _prepare_data_per_task(self, x_data, y_data, flatten_y=True):
        # a) make arrays 2-dimensional
        x_data, y_data = _handle_input_dimensionality(x_data, y_data)

        # b) normalize data
        x_data, y_data = self._normalize_data(x_data, y_data)

        if flatten_y:
            assert y_data.shape[1] == 1
            y_data = y_data.flatten()

        # c) convert to tensors
        x_tensor = torch.from_numpy(x_data).float().to(device)
        y_tensor = torch.from_numpy(y_data).float().to(device)

        return x_tensor, y_tensor

def _calib_error(pred_dist_vectorized, y_valid_tensor):
    cdf_vals = pred_dist_vectorized.cdf(y_valid_tensor)

    if y_valid_tensor.shape[0] == 1:
        y_valid_tensor = y_valid_tensor.flatten()
        cdf_vals = cdf_vals.flatten()

    num_points = y_valid_tensor.shape[0]
    conf_levels = torch.linspace(0.05, 0.95, 20).to(device)
    emp_freq_per_conf_level = torch.sum(cdf_vals[:, None] <= conf_levels, dim=0).float() / num_points

    calib_rmse = torch.sqrt(torch.mean((emp_freq_per_conf_level - conf_levels)**2))
    return calib_rmse