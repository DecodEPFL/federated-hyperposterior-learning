import torch, sys, os, copy
import numpy as np
import torch.nn.functional as F
from collections.abc import Iterable

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.GPR_meta_fedavg import load_serialized_fedavg_model
from config import device

def softplus_inverse(x):
    if isinstance(x, torch.Tensor):
        return x + torch.log(-torch.expm1(-x))
    else:
        x = torch.Tensor([x])
        return (x + torch.log(-torch.expm1(-x))).detach().numpy()[0]



def normalize_data_tup(data):
    # for one client only
    data_nrm=(None, None, None, None)
    # statistics on train data
    x_mean, y_mean = np.mean(data[0], axis=0), np.mean(data[1], axis=0)
    x_std, y_std = np.std(data[0], axis=0) + 1e-8, np.std(data[1], axis=0) + 1e-8
    # normalize
    data_nrm = ((data[0] - x_mean[None, :]) / x_std[None, :],
                (data[1] - y_mean[None, :]) / y_std[None, :],
                (data[2] - x_mean[None, :]) / x_std[None, :],
                (data[3] - y_mean[None, :]) / y_std[None, :])
    # check sizes
    assert data_nrm[0].shape == data[0].shape
    assert data_nrm[1].shape == data[1].shape
    assert data_nrm[2].shape == data[2].shape
    assert data_nrm[3].shape == data[3].shape

    return data_nrm, y_mean, y_std






# ---------- PRINT HYPER-PARAMETERS OF GP PRIOR ----------
import gpytorch
from server import GPRegressionLearned, GPRegressionMetaFedAvg, GPRegressionMetaLearnedSVGD, GPRegressionMetaLearnedVI

def print_gp_prior_params(gp, print_nn_weights=False):
    # main function to be used
    if isinstance(gp, GPRegressionLearned) or isinstance(gp, GPRegressionMetaFedAvg):
        return _print_gp_prior_params(gp)
    elif isinstance(gp, GPRegressionMetaLearnedSVGD):
        return gp.print_gp_prior_params(print_nn_weights=print_nn_weights)
    elif isinstance(gp, GPRegressionMetaLearnedVI):
        return ' ' #TODO
    else:
        raise NotImplementedError


def tensor_arr_to_str(x):
    return np.array2string(x.cpu().detach().numpy(), formatter={'float_kind':lambda x: "%.2f" % x})


def _print_gp_prior_params(gp):
    assert isinstance(gp, GPRegressionLearned) or isinstance(gp, GPRegressionMetaFedAvg)
    if isinstance(gp, GPRegressionLearned): # gp.model is a LearnedGPRegressionModel
        likelihood = gp.model.likelihood
        learned_kernel, learned_mean = gp.model.learned_kernel, gp.model.learned_mean,
        covar_module, mean_module = gp.model.covar_module, gp.model.mean_module
    elif isinstance(gp, GPRegressionMetaFedAvg):
        likelihood = gp.likelihood
        learned_kernel, learned_mean = gp.nn_kernel_map, gp.nn_mean_fn
        covar_module, mean_module = gp.covar_module, gp.mean_module
    else:
        raise NotImplementedError

    msg = ''
    # print kernel params
    if isinstance(covar_module, gpytorch.kernels.ScaleKernel): # SE or NN kernel
        if learned_kernel is None: # SE kernel
            msg += '\nSE kernel with lengthscale'
            msg += tensor_arr_to_str(covar_module.base_kernel.lengthscale)
            msg += 'raw = ' + tensor_arr_to_str(covar_module.base_kernel.raw_lengthscale)
            msg += '\nSE kernel with outputscale'
            msg += tensor_arr_to_str(covar_module.outputscale) + 'raw = '
            msg += tensor_arr_to_str(covar_module.raw_outputscale)
            msg += '\nSE kernel with noise'
            msg += tensor_arr_to_str(likelihood.noise) + 'raw = '
            msg += tensor_arr_to_str(likelihood.raw_noise)
        else: # NN kernel
            msg += '\nNN kernel with lengthscale '
            msg += tensor_arr_to_str(covar_module.base_kernel.lengthscale) + 'raw = '
            msg += tensor_arr_to_str(covar_module.base_kernel.raw_lengthscale)
            msg += '\nNN kernel with outputscale'
            msg += tensor_arr_to_str(covar_module.outputscale) + 'raw = '
            msg += tensor_arr_to_str(covar_module.raw_outputscale)
            msg += '\nNN kernel with noise'
            msg += tensor_arr_to_str(likelihood.noise) + 'raw = '
            msg += tensor_arr_to_str(likelihood.raw_noise)
            # print NN weights
            #for i in range(1, learned_kernel.n_layers+1):
            #    layer = getattr(learned_kernel, learned_kernel.prefix + 'fc_%i'%i) # hidden layer i
            #    print('kernel hidden layer {:2.0f} weights'.format(i), layer.weight)
            #    print('kernel hidden layer {:2.0f} bias'.format(i), layer.bias)
            #layer = getattr(learned_kernel, learned_kernel.prefix + 'out') # output layer
            #print('kernel output layer {:2.0f} weights'.format(i), layer.weight)
            #print('kernel output layer {:2.0f} bias'.format(i), layer.bias)
    if isinstance(covar_module, gpytorch.kernels.LinearKernel): # linear kernel
            msg += '\nLinear kernel with variance'
            msg += tensor_arr_to_str(covar_module.variance) + 'raw = '
            msg += tensor_arr_to_str(covar_module.raw_variance)
            msg += '\nLinear kernel with noise'
            msg += tensor_arr_to_str(likelihood.noise) + 'raw = '
            msg += tensor_arr_to_str(likelihood.raw_noise)
    # print mean params
    #if not learned_mean is None: # NN mean
    #    for i in range(1, learned_mean.n_layers+1):
    #        layer = getattr(learned_mean, learned_mean.prefix + 'fc_%i'%i) # hidden layer i
    #        print('NN hidden layer {:2.0f} weights'.format(i), layer.weight)
    #        print('NN hidden layer {:2.0f} bias'.format(i), layer.bias)
    #    layer = getattr(learned_mean, learned_mean.prefix + 'out') # output layer
    #    print('NN output layer {:2.0f} weights'.format(i), layer.weight)
    #    print('NN output layer {:2.0f} bias'.format(i), layer.bias)
    if isinstance(mean_module, gpytorch.means.ConstantMean):
        msg += '\nConstant mean = ' + tensor_arr_to_str(mean_module.constant)
    if isinstance(mean_module, gpytorch.means.LinearMean):
        msg += '\nLinear mean with weights' + tensor_arr_to_str(mean_module.weights)
        msg += '\nLinear mean with bias' + tensor_arr_to_str(mean_module.bias)

    return msg



 # ----------  ----------
def tile_in_list(a, l):
    '''
    a: values
    l: list of length of the output list
    if a is a list and l is a list, repeats a_i for l_i times.
    if a is a single value, repeats a for sum(l) times
    '''
    # convert all to list
    if not isinstance(a,list):
        a=[a]
    if not isinstance(l,list):
        l=[l]
    # total length of the result
    len_tot = sum(l)

    # if a is already the correct length, return it
    if len(a)==len_tot:
        return a
    # if a is a single element, repeat it
    if len(a)==1:
        return [a[0]]*len_tot
    # repeat a_i for l_i times
    if len(a)==len(l):
        res=[]
        for ai, li in zip(a, l):
            res += [ai]*li
        return res
    else:
        print('[ERROR]: should pass a single value, list of values to be tiled, or a full list')
        return []


def get_num_hyp_params(
    mean_nn_size, covar_nn_size, input_dim, feature_dim,
    optimize_noise=True, optimize_lengthscale=True):
    '''
    returns total number of hyper-parameters when both mean and kernel are NNs.
    Note: assumes a Gaussian likelihood with only 1 hyper-param (noise std)
    Note: for NN kernel, assumes an RBF without output scale and with 1 lengthscale per feature dim is applied on top
    - mean_nn_size, covar_nn_size: number of neurons at each hidden layer.
                                   set to None if mean or kernel is not a NN
    - input_dim: input dim
    - feature_dim: kernel output dim
    '''
    res = 0
    # noise std of the Gaussian likelihood
    if optimize_noise:
        res += 1
    # number of parameters in mean NN
    if not mean_nn_size is None:
        last_layer_size = input_dim
        mean_nn_size = mean_nn_size + (1,)                   # one output neuron
        for layer_num in np.arange(len(mean_nn_size)):
            res += last_layer_size*mean_nn_size[layer_num]  # weights coming to current layer
            res += mean_nn_size[layer_num]                  # biases of this layer
            last_layer_size = mean_nn_size[layer_num]
    # number of parameters in kernel NN
    if not covar_nn_size is None:
        last_layer_size = input_dim
        covar_nn_size = covar_nn_size + (feature_dim,)       # output neurons
        for layer_num in np.arange(len(covar_nn_size)):
            res += last_layer_size*covar_nn_size[layer_num] # weights coming to current layer
            if not layer_num == len(covar_nn_size)-1:
                res += covar_nn_size[layer_num]             # biases of 'hidden' layer
            last_layer_size = covar_nn_size[layer_num]
    #lengthscale per output dim
    if optimize_lengthscale:
        res += feature_dim
    return res


# ----- LOAD TRAINED MODELS -----
import pickle
from trained_svgd import TrainedSVGD
def load_trained_models(
    mode, filename_res, ts_data, methods=None,
    clients_subset=None, clients_train_data=None):

    if mode in ['vanilla', 'meta_fedavg']:
        assert not clients_train_data is None
    else:
        assert clients_train_data is None

    #if not filename_res.endswith(mode):
    #    filename_res = os.path.join(filename_res, mode)
    # find all methods if not specified
    if methods is None:
        methods = [f.split('_models')[0] for f in os.listdir(filename_res) if '_models' in f]
    if 'linreg' in methods:
        methods.remove('linreg')
    # convert to list and remove repeated methods
    methods = list(set(methods))
    # check inputs
    assert mode in ['personal', 'ours', 'meta_fedavg', 'vanilla']
    if (mode in ['vanilla', 'personal']) and (clients_subset is None):
        assert not clients_train_data is None, 'clients_subset or clients_train_data must be given'
        clients_subset = np.arange(len(clients_train_data))
    # init
    models_all = dict.fromkeys(methods)
    results_all = dict.fromkeys(methods)

    for method in methods:
        # --- 0. find latest models ---
        models_files = [f for f in os.listdir(filename_res) if f.startswith(method + '_models')]
        models_files = [os.path.join(filename_res, f) for f in models_files if os.path.isfile(os.path.join(filename_res, f))]
        if models_files==[]:
            results_all[method] = None
            print('\n[WARN] results of method ' + method + ' not found.')
            continue
        dates = np.array([int(f[-2:]) + 100*int(f[-5:-3]) for f in models_files])
        models_file = models_files[np.argmax(dates)]

        # --- 1. load models and results ---
        file = open(models_file, 'rb')
        res = pickle.load(file)
        file.close()
        print('\n[INFO] results of method '+ method + ' loaded.')
        # get results
        results_all[method] = res['results']


        # --- 2. reshape personal results ---
        if mode in ['vanilla', 'personal']:
            scenario_names = [x for x in results_all[method].keys() if not results_all[method][x] is None]
            for scenario_name in scenario_names:
                for criterion in results_all[method][scenario_name].keys():
                    list_of_dicts = copy.deepcopy(results_all[method][scenario_name][criterion])
                    results_all[method][scenario_name][criterion] = {
                        'criterion_train': [None]*len(list_of_dicts),
                        'criterion_valid': [None]*len(list_of_dicts),
                        'setup': [None]*len(list_of_dicts)
                    }
                    for ind, d in enumerate(list_of_dicts):
                        if not d is None:
                            for key in ['criterion_train', 'criterion_valid']:
                                results_all[method][scenario_name][criterion][key][ind] = d[key][0]
                            results_all[method][scenario_name][criterion]['setup'][ind] = d['setup']


        # --- 3. reconstruct models ---
        # get models
        if 'models' in res.keys():
            models_all[method] = res['models']
        else:
            print('[WARN] models trained by method '+ method + ' not found.')
            continue
        # reconstruct
        scenario_names = [x for x in models_all[method].keys() if not models_all[method][x] is None]
        for scenario_name in scenario_names:
            for criterion in models_all[method][scenario_name].keys():
                if mode == 'ours':
                    models_all[method][scenario_name][criterion] = TrainedSVGD(
                        models_all[method][scenario_name][criterion], ts_data=ts_data)
                elif mode=='meta_fedavg':
                    models_all[method][scenario_name][criterion] = load_serialized_fedavg_model(
                        clients_train_data,
                        models_all[method][scenario_name][criterion])
                else:
                    if isinstance(models_all[method][scenario_name][criterion], list):
                        if models_all[method][scenario_name][criterion][clients_subset[0]]==None:
                            print('[WARN] models trained by method '+ method + ' in scenario ' + scenario_name + ' not found.')
                            continue
                        else:
                            print('[INFO] models trained by method '+ method + ' in scenario ' + scenario_name + ' loaded.')
                    for client_num in clients_subset:
                        if mode=='vanilla':
                            models_all[method][scenario_name][criterion][client_num] = load_serialized_fedavg_model(
                                [clients_train_data[client_num]],
                                models_all[method][scenario_name][criterion][client_num])
                        else:
                            models_all[method][scenario_name][criterion][client_num] = TrainedSVGD(
                                models_all[method][scenario_name][criterion][client_num], ts_data=ts_data)

    return models_all, results_all



def fix_saved(mode, filename_res, methods=None, clients_subset=None):
    # find all methods if not specified
    if methods is None:
        methods = [f.split('_models')[0] for f in os.listdir(filename_res) if '_models' in f]

    # convert to lists
    methods = list(set(methods))
    # check inputs
    assert mode in ['personal', 'ours']
    if mode == 'personal':
        assert not clients_subset is None


    for method in methods:
        # --- 0. find latest models ---
        models_files = [f for f in os.listdir(filename_res) if f.startswith(method + '_models')]
        models_files = [os.path.join(filename_res, f) for f in models_files if os.path.isfile(os.path.join(filename_res, f))]
        if models_files==[]:
            results_method = None
            print('\n[WARN] results of method ' + method + ' not found.')
            continue
        dates = np.array([int(f[-2:]) + 100*int(f[-5:-3]) for f in models_files])
        models_file = models_files[np.argmax(dates)]

        # --- 1. load models and results ---
        file = open(models_file, 'rb')
        print(models_file)
        res = pickle.load(file)
        file.close()

        # get results
        results_method = res['results']
        # get models
        if 'models' in res.keys():
            models_method = res['models']
        else:
            print('[WARN] models trained by method '+ method + ' not found.')
            continue

        # --- 2. Apply the fix ---
        scenario_names = [x for x in models_method.keys() if not models_method[x] is None]
        for scenario_name in scenario_names:
            for criterion in models_method[scenario_name].keys():
                if models_method[scenario_name][criterion] is None:
                    print('[Err] method ' + method + ' scen ' + scenario_name + ' crit ' + criterion)
                    continue
                if mode == 'ours':
                    if 'fixl' in method:
                        models_method[scenario_name][criterion]['optimize_lengthscale'] = False
                        models_method[scenario_name][criterion]['lengthscale_fix'] = [[0.3, 0.3]]
                        if 'setup' in results_method[scenario_name][criterion].keys():
                            if 'lengthscale_fix' in results_method[scenario_name][criterion]['setup'].keys():
                                models_method[scenario_name][criterion]['lengthscale_fix'] = results_method[scenario_name][criterion]['setup']['lengthscale_fix']

                    else:
                        models_method[scenario_name][criterion]['optimize_lengthscale'] = True
                        models_method[scenario_name][criterion]['lengthscale_fix'] = None

                else:
                    if isinstance(models_method[scenario_name][criterion], list):
                        if models_method[scenario_name][criterion][clients_subset[0]]==None:
                            print('[WARN] models trained by method '+ method + ' in scenario ' + scenario_name + ' not found.')
                            continue
                        else:
                            print('[INFO] models trained by method '+ method + ' in scenario ' + scenario_name + ' loaded.')
                    for client_num in clients_subset:
                        if 'fixl' in method:
                            models_method[scenario_name][criterion][client_num]['optimize_lengthscale'] = False
                            models_method[scenario_name][criterion]['lengthscale_fix'][client_num] = [[0.3, 0.3]]
                            if 'setup' in results_method[scenario_name][criterion].keys():
                                if 'lengthscale_fix' in results_method[scenario_name][criterion]['setup'].keys():
                                    models_method[scenario_name][criterion]['lengthscale_fix'][client_num] = results_method[scenario_name][criterion]['setup'][client_num]['lengthscale_fix']
                        else:
                            models_method[scenario_name][criterion][client_num]['optimize_lengthscale'] = True
                            models_method[scenario_name][criterion][client_num]['lengthscale_fix'] = None

        # put back
        filename = models_file[:-2]+'17'
        with open(filename, "wb") as f:
            pickle.dump({'results':results_method, 'models':models_method}, f)


def calc_mean(arr, axis=None):
    # convert tuple to list
    if isinstance(arr, tuple):
        arr = list(arr)
    # calc mean
    if isinstance(arr, np.ndarray):
        return np.mean(arr, axis=axis)
    elif isinstance(arr, list):
        return sum(arr)/len(arr)
    elif not isinstance(arr, Iterable):
        return arr
    else:
        raise NotImplementedError



if __name__ == "__main__":
    print(get_num_hyp_params(
            mean_nn_size=None, covar_nn_size=(2, 2), input_dim=15, feature_dim=2))
    print(get_num_hyp_params(
            mean_nn_size=None, covar_nn_size=(2, 4, 2), input_dim=15, feature_dim=2))
    print(get_num_hyp_params(
            mean_nn_size=None, covar_nn_size=(2, 2, 2, 2), input_dim=15, feature_dim=2))
    print(get_num_hyp_params(
            mean_nn_size=None, covar_nn_size=(4, 4), input_dim=15, feature_dim=2))
    print(get_num_hyp_params(
            mean_nn_size=None, covar_nn_size=(4, 2, 2, 4), input_dim=15, feature_dim=2))
    print(get_num_hyp_params(
            mean_nn_size=None, covar_nn_size=(4, 3, 4), input_dim=15, feature_dim=2))
