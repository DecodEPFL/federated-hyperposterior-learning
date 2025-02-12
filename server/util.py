import warnings, copy
import numpy as np
from absl import flags
import warnings, torch
import os, sys, logging
import torch.nn.functional as F


def warning_format(message, category, filename, lineno, line=None):
    msg = '[WARNING] %s %s' % (message, category.__name__)
    msg += '\nin file' + filename + ' at line number:' + str(lineno)
    print(msg)
    return msg


def warning_show(message, category, filename, lineno, logger=None, line=None):
    msg = '[WARNING] '
    msg += '%s:%s: %s:%s' % (filename, lineno, category.__name__, message)
    if not (isinstance(logger, WrapLogger) or logger==None):
        logger=WrapLogger(logger)
    logger.info(msg)
    return


def find_root_by_bounding(fun, left, right, eps=1e-6, max_iter=1e4):
    """
    Root finding method that uses selective shrinking of a target interval bounded by left and right
    --> other than the newton method, this method only works for for vectorized univariate functions
    Args:
        fun (callable): function f for which f(x) = 0 shall be solved
        left: (torch.Tensor): initial left bound
        right (torch.Tensor): initial right bound
        eps (float): tolerance
        max_iter (int): maximum iterations
    """

    assert callable(fun)

    n_iter = 0
    approx_error = 1e12
    while approx_error > eps:
        middle = (right + left)/2
        f = fun(middle)

        left_of_zero = (f < 0).flatten()
        left[left_of_zero] = middle[left_of_zero]
        right[~left_of_zero] = middle[~left_of_zero]

        assert torch.all(left <= right).item()

        approx_error = torch.max(torch.abs(right-left))/2
        n_iter += 1

        if n_iter > max_iter:
            warnings.warn("Max_iter has been reached - stopping newton method for determining quantiles")
            return torch.tensor([np.nan for _ in range(len(left))] )

    return middle

def _handle_input_dimensionality(x, y=None):
    if x.ndim == 1:
        x = np.expand_dims(x, -1)

    assert x.ndim == 2

    if y is not None:
        if y.ndim == 1:
            y = np.expand_dims(y, -1)
        assert x.shape[0] == y.shape[0]
        assert y.ndim == 2

        return x, y
    else:
        return x


def get_logger(log_dir=None, log_file='output.log', expname=''):
    # change red background
    from IPython.core.display import HTML
    HTML("""
    <style>
    div.output_stdout {
        background: #0C0;
    }
    </style>
    """)

    if log_dir is None and flags.FLAGS.is_parsed() and hasattr(flags.FLAGS, 'log_dir'):
        log_dir = flags.FLAGS.log_dir

    logger = logging.getLogger('gp-priors')
    logger.setLevel(logging.INFO)

    if len(logger.handlers) == 0:

        #formatting
        if len(expname) > 0:
            expname = ' %s - '%expname
        formatter = logging.Formatter('[%(asctime)s -' + '%s'%expname +  '%(levelname)s]  %(message)s')

        # Stream Handler
        sh = logging.StreamHandler(sys.stdout) # NOTE: can remove arg
        sh.setFormatter(formatter)
        sh.setLevel(logging.INFO)
        logger.addHandler(sh)

        logger.propagate = False

        # File Handler
        if log_dir is not None and len(log_dir) > 0:
            fh = logging.FileHandler(os.path.join(log_dir, log_file))
            fh.setFormatter(formatter)
            fh.setLevel(logging.INFO)
            logger.addHandler(fh)
            logger.log_dir = log_dir
        else:
            logger.log_dir = None
    return logger


class WrapLogger():
    def __init__(self, logger, verbose=True):
        self.can_log = not (logger == None)
        self.logger=logger
        self.verbose = verbose

    def info(self, msg):
        if self.can_log:
            self.logger.info(msg)
        if self.verbose:
            print(msg)

    def close(self):
        if not self.can_log:
            return
        while len(self.logger.handlers):
            h = self.logger.handlers[0]
            h.close()
            self.logger.removeHandler(h)


class DummyLRScheduler:

    def __init__(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass





# ---- print gp params ------
def tensor_arr_to_str(x):
    return str(np.array2string(x.cpu().detach().numpy(), formatter={'float_kind':lambda x: "%.2f" % x}))


def print_gpvec_prior_params(gpvec, print_nn_weights=False):
    #assert isinstance(gpvec, server.random_gp.VectorizedGP)
    msg = ''
    # print covar module
    if (gpvec.covar_module_str=='NN') or (gpvec.covar_module_str=='SNN'):
        l_raw = gpvec.lengthscale_raw
        o_raw = gpvec.outputscale_raw
        msg += 'NN kernel with lengthscale = ' + tensor_arr_to_str(F.softplus(l_raw))
        msg += ' (raw = ' +  str(tensor_arr_to_str(l_raw)) + ')'
        msg += '\nNN kernel with outputscale = ' + tensor_arr_to_str(F.softplus(o_raw))
        msg += ' (raw = ' +  str(tensor_arr_to_str(o_raw)) + ')'
        # kernel NN weight
        for i in np.arange(gpvec.kernel_nn.n_layers):
            weights = getattr(gpvec.kernel_nn, 'fc_%i'%(i+1))
            msg += '\nnorm of weights in hidden layer {:1.0f}'.format(i)
            msg += ' = ' + tensor_arr_to_str(torch.norm(weights.weight.flatten(), p=2))
            msg += ', norm of biases'
            msg += ' = ' + tensor_arr_to_str(torch.norm(weights.bias.flatten(), p=2))
            if print_nn_weights:
                msg += 'kernel_nn fc_%i'%(i+1) + ' weights: ' + tensor_arr_to_str(weights.weight)
                msg += 'kernel_nn fc_%i'%(i+1) + ' bias: ' + tensor_arr_to_str(weights.bias)
        # output layer
        weights = getattr(gpvec.kernel_nn, 'out')
        msg += '\nnorm of weights in output layer'
        msg += ' = ' + tensor_arr_to_str(torch.norm(weights.weight.flatten(), p=2))
        msg += ', norm of biases'
        msg += ' = ' + tensor_arr_to_str(torch.norm(weights.bias.flatten(), p=2))
        if print_nn_weights:
            msg += 'kernel_nn output weights: ' + tensor_arr_to_str(weights.weight)
            msg += 'kernel_nn output bias: ' + tensor_arr_to_str(weights.bias)
    elif gpvec.covar_module_str == 'SE':
        l_raw = gpvec.lengthscale_raw
        o_raw = gpvec.outputscale_raw
        msg += 'SE kernel with lengthscale = ' + tensor_arr_to_str(F.softplus(l_raw))
        msg += ' (raw = ' +  tensor_arr_to_str(l_raw) + ')'
        msg += '\nSE kernel with outputscale = ' + tensor_arr_to_str(F.softplus(o_raw))
        msg += ' (raw = ' +  tensor_arr_to_str(o_raw) + ')'
    elif gpvec.covar_module_str == 'linear':
        msg += 'Linear kernel with variance raw = '  + tensor_arr_to_str(gpvec.variance_raw)
    elif gpvec.covar_module_str == 'zero':
        msg += 'Zero kernel '
    else:
        raise NotImplementedError

    # print mean params
    if gpvec.mean_module_str == 'NN':
        msg += '\nNN mean'
        # mean NN weight
        for i in np.arange(gpvec.mean_nn.n_layers):
            weights = getattr(gpvec.mean_nn, 'fc_%i'%(i+1))
            msg += '\nnorm of weights in hidden layer {:1.0f}'.format(i)
            msg += ' = ' + tensor_arr_to_str(torch.norm(weights.weight.flatten(), p=2))
            msg += ', norm of biases'
            msg += ' = ' + tensor_arr_to_str(torch.norm(weights.bias.flatten(), p=2))
            if print_nn_weights:
                msg += 'mean_nn fc_%i'%(i+1) + ' weights: ' + tensor_arr_to_str(weights.weight)
                msg += 'mean_nn fc_%i'%(i+1) + ' bias: ' + tensor_arr_to_str(weights.bias)
        # output layer
        weights = getattr(gpvec.mean_nn, 'out')
        msg += '\nnorm of weights in output layer'
        msg += ' = ' + tensor_arr_to_str(torch.norm(weights.weight.flatten(), p=2))
        msg += ', norm of biases'
        msg += ' = ' + tensor_arr_to_str(torch.norm(weights.bias.flatten(), p=2))
        if print_nn_weights:
            msg += 'mean_nn output weights: ' + tensor_arr_to_str(weights.weight)
            msg += 'mean_nn output bias: ' + tensor_arr_to_str(weights.bias)

    elif gpvec.mean_module_str in ['zero', 'constant']:
        msg += '\nconstant mean = ' + tensor_arr_to_str(gpvec.constant_mean)
    elif gpvec.mean_module_str == 'linear_no_bias':
        msg +='\nlinear no bias mean weigths = ' + tensor_arr_to_str(gpvec.weights)
    else:
        raise NotImplementedError

    # print noise
    noise_std = gpvec.noise_raw.detach().clone().cpu()
    msg += '\nTuned noise std = ' + str(F.softplus(noise_std)) + ', raw = ' + str(noise_std)

    return msg


def softplus_inverse(x):
    if isinstance(x, torch.Tensor):
        return x + torch.log(-torch.expm1(-x))
    else:
        x = torch.tensor([x])
        return (x + torch.log(-torch.expm1(-x))).detach().numpy()[0]