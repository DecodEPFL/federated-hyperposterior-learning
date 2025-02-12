# ----- Import packages -----
import pickle, os, sys, logging, torch
import numpy as np
from datetime import datetime

sys.path.append(os.path.realpath(os.path.dirname(__file__))+'/../')

from utils.search_models import *
from ridge_wrapper import RidgeWrapper
from utils.trained_svgd import serialize_model

def train_models(exp_name, method_to_run, env_dict,
                 random_seed, criteria, options,
                 mode, pf_scheduler=None, train_scenarios=None,
                 save_models=True, initial_particles=None,
                 verbose=True, clients_subset=None,
                 file_name=None, approx_method='SVGD'):
    '''
    trains models, logs results, and saves the trained models.
    - exp_name: used to save models and log
    - method_to_run:
    - env_dict
    - train_scenarios: a subset of scenarios in env_dict.
                       if not specified, runs for all of them.
    '''
    assert approx_method in ['SVGD', 'VI', 'None']

    if mode=='ours' and method_to_run=='linreg':
        raise NotImplementedError
    if mode=='ours':
        assert not approx_method is None
        assert clients_subset==None #TODO

    if not isinstance(criteria, list):
        criteria = [criteria]

    #assert method_to_run in ['linreg', 'lingp', 'nn4', 'nn16', 'res_gp', 'res_gp']
    msg = '\n----------[INFO] running '+ method_to_run + ' ----------'

    if train_scenarios is None:
        train_scenarios = copy.deepcopy(env_dict['train_scenarios'])
    else:
        if not isinstance(train_scenarios, list):
            train_scenarios = [train_scenarios]
        train_scenarios = {k: copy.deepcopy(env_dict['train_scenarios'][k]) for k in train_scenarios}
    assert set(train_scenarios).issubset(set(env_dict['train_scenarios']))

    if not options[method_to_run]['normalize_data']:
        msg += '\n[WARNING] normalize inputs to avoid numerical issues in calculating GP cov inverse.\n'
    if clients_subset is None:
        clients_subset = np.arange(env_dict['num_clients'])
        msg += '\n[INFO] Running for all clients'
    else:
        msg += '\n[INFO] Subset of clients for demonstration: '+ str(clients_subset)


    # ----- SET UP LOGGER -----
    now = datetime.now().strftime("%m_%d")
    file_name_base = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
    file_name_base = os.path.join(file_name_base, 'experiments', 'PV', 'saved_results', exp_name)
    if file_name is None:
        filename_log  = os.path.join(file_name_base, mode, method_to_run) + "_log" + now
        filename_save = os.path.join(file_name_base, mode, method_to_run) + "_models" + now
    else:
        filename_log  = os.path.join(file_name_base, file_name + "_log" + now)
        filename_save = os.path.join(file_name_base, file_name + "_models" + now)
    logging.basicConfig(filename=filename_log, format='%(asctime)s %(message)s', filemode='w')
    logger=logging.getLogger(method_to_run)
    logger.setLevel(logging.DEBUG)
    logger = WrapLogger(logger)


    # ----- FEATURES SUBSET -----
    if 'features_subset' in options[method_to_run].keys():
        # selecting a subset of features given by their names
        if len(options[method_to_run]['features_subset'])<len(env_dict['feature_names']):
            cols_subset = [i for i,x in enumerate(env_dict['feature_names']) if x in options[method_to_run]['features_subset']]
            msg += '\n[INFO] using features subset:' + ', '.join(options[method_to_run]['features_subset'])
            for scenario_name in train_scenarios.keys():
                clients_data = train_scenarios[scenario_name]['clients_data']
                for client_num in clients_subset:
                    # data with all features
                    x_train, y_train, x_valid, y_valid = clients_data[client_num]
                    # data with reduced features
                    clients_data[client_num] = x_train[:, cols_subset], y_train, x_valid[:, cols_subset], y_valid
        else:
            cols_subset = np.arange(len(env_dict['feature_names']))
            msg += '\n[INFO] using all features:' + ', '.join(env_dict['feature_names'])
        del options[method_to_run]['features_subset']
    else:
        cols_subset = np.arange(len(env_dict['feature_names']))
        msg += '\n[INFO] using all features:' + ', '.join(env_dict['feature_names'])


    # --- PRIOR FACTOR ___
    # NOTE: would be more precise to set by num samples of each client in personal mode
    if method_to_run=='linreg' or mode in ['vanilla', 'meta_fedavg']:
        options[method_to_run]['prior_factor'] = 0
        pf_scheduler = None
    else:
        if 'prior_factor' in options[method_to_run].keys():
            if not pf_scheduler is None:
                msg += '\n[WARN] prior factor scheduler not used.'
                pf_scheduler = None
        elif not pf_scheduler is None:
            msg += '\n[INFO] automatically setting prior factors.'
            if 'prior_factor' in options[method_to_run].keys():
                del options[method_to_run]['prior_factor']
        else:
            msg += '[ERR] prior factor scheduler or prior factor must be given.'
            options[method_to_run]['prior_factor'] = [0]


    logger.info(msg)


    # ----- TRAIN -----
    models = dict.fromkeys(train_scenarios.keys())
    results = dict.fromkeys(train_scenarios.keys())
    for scenario_name in train_scenarios.keys():
        logger.info('\n[INFO] training scenario: ' + scenario_name)

        clients_data = train_scenarios[scenario_name]['clients_data']

        # TRAIN
        if mode in ['personal', 'vanilla']:
            models[scenario_name], results[scenario_name] = _train_models_indiv(
                clients_data=clients_data, approx_method=approx_method,
                method_to_run=method_to_run, criteria=criteria,
                random_seed=random_seed, pf_scheduler=pf_scheduler,
                options=options, initial_particles=initial_particles,
                logger=logger, verbose=verbose,
                # inputs specific to personal
                clients_subset=clients_subset,
                num_clients=env_dict['num_clients'], mode=mode
            )
        # FL methods
        elif mode in ['meta_fedavg', 'ours']:
            models[scenario_name], results[scenario_name] = _train_models_fl(
                clients_data=clients_data, approx_method=approx_method,
                method_to_run=method_to_run, criteria=criteria,
                random_seed=random_seed, pf_scheduler=pf_scheduler,
                options=options, initial_particles=initial_particles,
                logger=logger, verbose=verbose, mode=mode
            )
        else:
            raise NotImplementedError


    # SAVE
    if save_models:
        file = open(filename_save, 'wb')
        if not method_to_run=='linreg':
            pickle.dump({'results': results, 'models':models}, file)
        else:
            pickle.dump({'results': results}, file)
        file.close()
    # close logger
    logger.close()
    del logger

    return models, results




# ------------------------------------------------
def _train_models_indiv(
        clients_subset, clients_data,
        method_to_run, pf_scheduler,
        criteria, random_seed, approx_method,
        options, initial_particles, num_clients,
        logger, verbose, mode):

    # ----- INIT -----
    models = [None]*num_clients
    results = [None]*num_clients

    # train for each clients
    for client_num in clients_subset:
        logger.info('\nClient {:2.0f}'.format(client_num))

        if method_to_run=='linreg':
            models[client_num], results[client_num] = best_lin_reg(
                clients_data=clients_data, client_num_fix=client_num, criterion=criteria[0],
                logger=logger, normalize_data=options[method_to_run]['normalize_data'], verbose=verbose)
        else:
            # vanilla = meta_fedavg with 1 client and personal = ours with 1 client
            mode_tmp = 'meta_fedavg' if mode=='vanilla' else 'ours'
            models[client_num], results[client_num] = find_best_model(
                clients_data=[clients_data[client_num]], client_num_fix=None,
                criteria=criteria, random_seed=random_seed, logger=logger,
                mode=mode_tmp, verbose=True, options=options[method_to_run],
                pf_scheduler=pf_scheduler, initial_particles=initial_particles,
                approx_method=approx_method)

    # reshape results: from list of fields to {field: list}
    fields = list(results[clients_subset[0]].keys())
    results_new = dict.fromkeys(fields)
    for field in fields:
        # init
        results_new[field] = [None]*num_clients
        # fill for each client
        for client_num in clients_subset:
            tmp_res = copy.deepcopy(results[client_num][field])
            if not isinstance(tmp_res, list):
                results_new[field][client_num] = tmp_res
            else:
                if len(tmp_res)==1:
                    results_new[field][client_num] = tmp_res[0]
                else:
                    raise NotImplementedError



    # compress and reshape models
    if not (isinstance(models[client_num], Ridge) or isinstance(models[client_num], RidgeWrapper) or isinstance(models[client_num], LinearRegression)):
        models_srz = dict.fromkeys(criteria)
        for criterion in criteria:
            models_srz[criterion] = [None]*num_clients
            for client_num in clients_subset:
                if mode=='personal':
                    models_srz[criterion][client_num] = serialize_model(
                        models[client_num][criterion],
                        normalize_data=options[method_to_run]['normalize_data'],
                        random_seed=random_seed)
                elif mode=='vanilla':
                    models_srz[criterion][client_num] = models[client_num][criterion].serialize_model()

    else:
        models_srz = None
    return models_srz, results_new


# ------------------------------------------------
def _train_models_fl(
        clients_data, approx_method,
        method_to_run, options, pf_scheduler,
        criteria, random_seed, initial_particles,
        logger, verbose, mode):

    if method_to_run=='linreg':
        models, results = best_lin_reg(
                clients_data=clients_data, client_num_fix=None, criterion=criteria[0],
                logger=logger, normalize_data=options[method_to_run]['normalize_data'], verbose=verbose)
        return models, results


    models, results = find_best_model(
            clients_data=clients_data, client_num_fix=None, pf_scheduler=pf_scheduler,
            criteria=criteria, random_seed=random_seed, logger=logger, initial_particles=initial_particles,
            mode=mode, verbose=True, options=options[method_to_run], approx_method=approx_method)

    # COMPRESS MODELS
    models_srz = dict.fromkeys(criteria)
    for criterion in criteria:
        if mode=='ours':
            models_srz[criterion] = serialize_model(
                        models[criterion],
                        normalize_data=options[method_to_run]['normalize_data'],
                        random_seed=random_seed)
        elif mode=='meta_fedavg':
            models_srz[criterion] = models[criterion].serialize_model()
    return models_srz, results


if __name__ == "__main__":
    exp_name='PV_test'
    save_models = True
    verbose=False
    criteria = ['rsmse']

    # -------- Define a fixed hyper-prior --------
    hyper_prior_dict = {'lengthscale_raw_loc': 5,'lengthscale_raw_scale': 2.5,
                        'noise_raw_loc': 10, 'noise_raw_scale': 5}

    # set properties that are different based on the GP mean
    options_base={
            # noise setup
            'optimize_noise':True, 'noise_std':None,
            # linear model
            'covar_module_str': ['zero'], 'mean_module_str':['linear'],
            'kernel_nn_layers': [], 'mean_nn_layers':[],
            'nonlinearity_output_m': None, 'nonlinearity_output_k': None,
            'nonlinearity_hidden_m': None, 'nonlinearity_hidden_k': None,
            'feature_dim':None,
            'optimize_lengthscale':True,
            'lengthscale_fix':None,
            # Configuration for GP-Prior learning
            'lr': 1e-2,           # learning rate for Adam optimizer'
            'lr_decay': 0.90,     # 'multiplicative learning rate decay factor applied after every 1000 steps'
            'task_batch_size': 5, # 'batch size for meta training, i.e. number of tasks for computing grads'
            'normalize_data': True,
            # Configuration for SVGD
            'num_iter_fit': 2000,
            'max_iter_fit': 3000,
            'early_stopping': True,
            'num_particles': [1],
            'bandwidth': 1,
            'n_threads': 8,
            'hyper_prior_dict': hyper_prior_dict,
            # loss parameters
            'prior_factor': None
    }

    options = {
            'lingp': copy.deepcopy(options_base),
            'nn4': copy.deepcopy(options_base),
            'nn16': copy.deepcopy(options_base)
            }
    options['nn4'].update({
                    'covar_module_str': ['zero'], 'mean_module_str':['NN'],
                    'kernel_nn_layers': [], 'mean_nn_layers':[(4,4)],
                    'nonlinearity_output_m': None, 'nonlinearity_hidden_m': [torch.relu, torch.tanh],
                    'feature_dim':None,
                    'num_iter_fit': 4000,
                    'max_iter_fit': 6000})
    options['nn16'].update({
                    'covar_module_str': ['zero'], 'mean_module_str':['NN'],
                    'kernel_nn_layers': [], 'mean_nn_layers':[(16,16)],
                    'nonlinearity_output_m': None, 'nonlinearity_hidden_m': [torch.relu, torch.tanh],
                    'feature_dim':None,
                    'num_iter_fit': 4000,
                    'max_iter_fit': 6000})

    filename_env  = os.path.realpath(os.path.dirname(os.path.dirname(__file__))) + "/experiments/PV/saved_results/PV_UniModal_env"
    print('--------')
    print(filename_env)
    print('--------')
    file = open(filename_env, 'rb')
    env_dict = pickle.load(file)
    msg = '[INFO] loaded data for {:2.0f} clients'.format(env_dict['num_clients'])
    print(msg)
    file.close()

    methods_to_run = ['linreg']
    for method_to_run in methods_to_run:
        train_models(exp_name=exp_name, method_to_run=method_to_run,
                    env_dict=env_dict, random_seed=1, criteria=criteria,
                    options=options, save_models=True,
                    verbose=True, clients_subset=[1])