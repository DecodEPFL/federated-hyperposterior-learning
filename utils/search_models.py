import numpy as np
import torch, itertools, sys, os, copy
from sklearn.linear_model import LinearRegression, Ridge
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.assistive_functions import print_gp_prior_params, calc_mean
from server.util import WrapLogger
from server import GPRegressionLearned, GPRegressionMetaLearnedSVGD, GPRegressionMetaLearnedVI, GPRegressionMetaFedAvg


# --------------- LINEAR AND RIDGE REGRESSION WITHOUT GP ---------------
def best_lin_reg(clients_data, client_num_fix, logger, criterion, normalize_data=True, verbose=True, alphas=None):
    assert criterion in ['rmse', 'rsmse']
    alphas = np.logspace(-6,2,80) if alphas is None else alphas
    best_res = dict.fromkeys(['train_rmse', 'valid_rmse', 'train_rsmse', 'valid_rsmse', 'alpha'])
    cur_res = dict.fromkeys(best_res.keys())
    x_train, y_train, x_valid, y_valid = clients_data[client_num_fix]

    if not isinstance(logger, WrapLogger):
        logger = WrapLogger(logger, verbose=verbose)

    # normalize data
    if normalize_data:
        # statistics on train data
        x_mean, y_mean = np.mean(x_train, axis=0), np.mean(y_train, axis=0)
        x_std, y_std = np.std(x_train, axis=0) + 1e-8, np.std(y_train, axis=0) + 1e-8
        # statistics on valid data
        y_valid_std = np.std(y_valid, axis=0) + 1e-8
        # normalize
        x_train_nrm = (x_train - x_mean[None, :]) / x_std[None, :]
        y_train_nrm = (y_train - y_mean[None, :]) / y_std[None, :]
        x_valid_nrm = (x_valid - x_mean[None, :]) / x_std[None, :]
        y_valid_nrm = (y_valid - y_mean[None, :]) / y_std[None, :]
        # check sizes
        assert x_train_nrm.shape == x_train.shape
        assert y_train_nrm.shape == y_train.shape
        assert x_valid_nrm.shape == x_valid.shape
        assert y_valid_nrm.shape == y_valid.shape
        x_train, y_train = x_train_nrm, y_train_nrm
        x_valid, y_valid = x_valid_nrm, y_valid_nrm
    else:
        x_mean, y_mean = np.zeros(x_train.shape[1]), np.zeros(y_train.shape[1])
        x_std, y_std = np.ones(x_train.shape[1]), np.ones(y_train.shape[1])


    # ------ non-regularized lin reg ------
    # train
    best_model = LinearRegression().fit(x_train, y_train)
    # evaluate
    y_train_pred = best_model.predict(x_train).reshape(y_train.shape)
    y_valid_pred = best_model.predict(x_valid).reshape(y_valid.shape)
    cur_res['valid_rmse'] = ((np.mean((y_valid_pred-y_valid)**2))**0.5)*y_std[0]
    cur_res['train_rmse'] = ((np.mean((y_train_pred-y_train)**2))**0.5)*y_std[0]
    cur_res['valid_rsmse'] = cur_res['valid_rmse']/y_valid_std[0]
    cur_res['train_rsmse'] = cur_res['train_rmse']/y_std[0]
    # dispaly
    msg = '\nNon-regularized linear model errors:'
    msg += '\nTrain R2 score = {:0.4f}, Valid R2 score = {:0.4f}'.format(best_model.score(x_train, y_train),
                                                                         best_model.score(x_valid, y_valid))
    msg += '\nTrain ' + criterion + ' = {:2.4f}'.format(cur_res['train_'+criterion])
    msg += ', Valid ' + criterion + ' = {:2.4f}'.format(cur_res['valid_'+criterion])
    best_res = copy.deepcopy(cur_res)
    best_res['alpha'] = 0
    logger.info(msg)

    # ------ regularized lin reg ------
    for alpha in alphas:
        # train
        reg = Ridge(alpha=alpha).fit(x_train, y_train)
        # evaluate
        y_valid_pred = reg.predict(x_valid).reshape(y_valid.shape)
        y_train_pred = reg.predict(x_train).reshape(y_train.shape)
        cur_res['valid_rmse'] = ((np.mean((y_valid_pred-y_valid)**2))**0.5)*y_std[0]
        cur_res['train_rmse'] = ((np.mean((y_train_pred-y_train)**2))**0.5)*y_std[0]
        cur_res['valid_rsmse'] = cur_res['valid_rmse']/y_valid_std[0]
        cur_res['train_rsmse'] = cur_res['train_rmse']/y_std[0]
        # compare with the best
        if cur_res['valid_'+criterion]< best_res['valid_'+criterion]:
            best_model = copy.deepcopy(reg)
            best_res = copy.deepcopy(cur_res)
            best_res['alpha'] = alpha


    # evauate the model with the lowest validation RMSE
    msg = '\nBest Ridge linear model (alpha = {:2.2f}):'.format(best_res['alpha'])
    msg += '\nTrain R2 score = {:0.4f}, Valid R2 score = {:0.4f}'.format(best_model.score(x_train, y_train),
                                                                        best_model.score(x_valid, y_valid))
    msg += '\nTrain ' + criterion + ' = {:2.4f}'.format(best_res['train_'+criterion])
    msg += ', Valid ' + criterion + ' = {:2.4f}'.format(best_res['valid_'+criterion])
    logger.info(msg)
    return best_model, best_res




# ---------------  TUNE DIFFERENT METHODS ---------------
def find_best_model(
    clients_data, random_seed, logger, criteria, approx_method='SVGD',
    pf_scheduler=None, mode='ours', client_num_fix=None, verbose=True,
    options={}, initial_particles={}, log_period=250):

    assert mode in ['personal', 'global', 'fedavg', 'ours', 'meta_fedavg', 'vanilla']
    assert approx_method in ['SVGD', 'VI']

    if not isinstance(criteria, list):
        criteria = [criteria]
    criteria = [criterion.lower() for criterion in criteria]
    for criterion in criteria:
        assert criterion in ['rmse', 'rsmse', 'calibr', 'nll']
    if not isinstance(logger, WrapLogger):
        logger = WrapLogger(logger, verbose=verbose)

    # ----- TRAIN DATA -------
    msg = '\n' + mode +' mode '
    if mode =='personal':
        assert not client_num_fix is None, "select client by passing client_num_fix"
        num_clients = 1
        msg += 'for client {:2.0f}'.format(client_num_fix)
        x_train, y_train, x_valid, y_valid = clients_data[client_num_fix]
    else:
        assert client_num_fix is None, "FL or global work on all clients."
        num_clients = len(clients_data)
        if mode in ['global', 'fedavg']:
            x_train, y_train, x_valid, y_valid = clients_data[0]
            for n in np.arange(1, num_clients):
                x_tr, y_tr, x_va, y_va = clients_data[n]
                x_train = np.concatenate((x_train, x_tr), axis=0)
                y_train = np.concatenate((y_train, y_tr), axis=0)
                x_valid = np.concatenate((x_valid, x_va), axis=0)
                y_valid = np.concatenate((y_valid, y_va), axis=0)
        elif mode in ['ours', 'meta_fedavg']:
            clients_train_data = [None]*num_clients
            for n in np.arange(num_clients):
                x_obs, y_obs, _, _ = clients_data[n]
                clients_train_data[n] = (x_obs, y_obs)
            del n

    # ------ parse inputs ------
    msg += '\n[INFO] finding the best model according to '.join([criterion for criterion in criteria])
    msg_in, lcls, setups, setup_keys = _parse_inputs(options=options, mode=mode)
    msg += '\n' + msg_in

    # ------ log common info ------ #
    logger.info(msg)
    if setups is None:
        raise NotImplementedError(msg)


    # ----- TRAIN MODELS -----
    best_gp = dict.fromkeys(criteria, None)
    best_res_all = dict.fromkeys(criteria, None)
    for criterion in criteria:
        best_res_all[criterion] = {
            'criterion_valid': [1e6]*num_clients,
            'criterion_train': [1e6]*num_clients,
            'setup':None # same for all clients
        }

    for model_ind, setup in enumerate(setups):
        torch.set_num_threads(lcls['n_threads'])

        # set variable options up and log
        msg_info = '\n[INFO]'
        # convert all variable info to variables
        for key, value in zip(setup_keys, setup):
            if key == 'nn_layers':
                lcls['mean_nn_layers'] = value
                lcls['kernel_nn_layers'] = value
                msg_info += 'mean and kernel NN: {:2.0f} x{:2.0f}'.format(value[0], value[1]) + ', '
            else:
                lcls[key] = value
                if key.startswith('nonlinearity') and (not value is None):
                    msg_info += key + ': ' + str(value.__name__) + ', '
                else:
                    msg_info += key + ': ' + str(value) + ', '


        # initialize pf scheduler
        if 'prior_factor' in lcls.keys() or pf_scheduler is None:
            pf_scheduler = None
        else:
            pf_scheduler.reset()
        done = False


        while not done:
            # 0. get prior factor
            if pf_scheduler is None: # given prior factors, fixed or list
                prior_factor = lcls['prior_factor']
            else: # automatic search over prior factor
                if pf_scheduler.done:
                    break
                prior_factor = pf_scheduler.get_pf()

            msg = msg_info + 'prior factor: {:1.6f}'.format(prior_factor)
            logger.info(msg)

            # 1. define model
            if mode in ['personal', 'global']:
                gp = GPRegressionLearned(
                            x_train, y_train, learning_mode='both',
                            num_iter_fit=lcls['num_iter_fit'],
                            covar_module=lcls['covar_module_str'],  #TODO: rename
                            mean_module=lcls['mean_module_str'],    #TODO: rename
                            kernel_nn_layers=lcls['kernel_nn_layers'],
                            mean_nn_layers=lcls['mean_nn_layers'],
                            optimizer='Adam', lr=lcls['lr'],
                            normalize_data=lcls['normalize_data'],
                            optimize_noise=lcls['optimize_noise'],
                            noise_std=lcls['noise_std'],
                            optimize_lengthscale=lcls['optimize_lengthscale'],
                            lengthscale_fix=lcls['lengthscale_fix'],
                            random_seed=random_seed,
                            nonlinearity_hidden_m=lcls['nonlinearity_hidden_m'],
                            nonlinearity_hidden_k=lcls['nonlinearity_hidden_k'],
                            nonlinearity_output_m=lcls['nonlinearity_output_m'],
                            nonlinearity_output_k=lcls['nonlinearity_output_k'],
                            weight_decay=0, # TODO
                            lr_scheduler=True, # TODO
                            ts_data=lcls['ts_data'])
            elif mode=='ours':
                if approx_method=='SVGD':
                    gp = GPRegressionMetaLearnedSVGD(
                        clients_train_data,
                        num_iter_fit=lcls['num_iter_fit'],
                        feature_dim=lcls['feature_dim'],            # only in ours #TODO
                        prior_factor=prior_factor,                  # only in ours
                        hyper_prior_dict=lcls['hyper_prior_dict'],  # only in ours
                        covar_module_str=lcls['covar_module_str'],
                        mean_module_str=lcls['mean_module_str'],
                        nonlinearity_hidden_m=lcls['nonlinearity_hidden_m'],
                        nonlinearity_hidden_k=lcls['nonlinearity_hidden_k'],
                        nonlinearity_output_m=lcls['nonlinearity_output_m'],
                        nonlinearity_output_k=lcls['nonlinearity_output_k'],
                        likelihood_str=lcls['likelihood_str'],      # only in ours #TODO
                        kernel_nn_layers=lcls['kernel_nn_layers'],
                        mean_nn_layers=lcls['mean_nn_layers'],
                        optimizer='Adam', lr=lcls['lr'],
                        lr_decay=lcls['lr_decay'],                  # only in ours #TODO
                        task_batch_size=lcls['task_batch_size'],    # only in ours
                        normalize_data=lcls['normalize_data'],
                        optimize_noise=lcls['optimize_noise'],
                        noise_std=lcls['noise_std'],
                        optimize_lengthscale=lcls['optimize_lengthscale'],
                        lengthscale_fix=lcls['lengthscale_fix'],
                        bandwidth=lcls['bandwidth'], kernel='RBF',  # only in ours SVGD
                        num_particles=lcls['num_particles'],        # only in ours SVGD
                        initial_particles=initial_particles,        # only in ours SVGD
                        random_seed=random_seed,
                        logger=logger,                              # only in ours #TODO
                        ts_data=lcls['ts_data']
                    )

                elif approx_method=='VI':
                    gp = GPRegressionMetaLearnedVI(
                        clients_train_data,
                        num_iter_fit=lcls['num_iter_fit'],
                        feature_dim=lcls['feature_dim'],            # only in ours #TODO
                        prior_factor=prior_factor,                  # only in ours
                        hyper_prior_dict=lcls['hyper_prior_dict'],  # only in ours
                        covar_module_str=lcls['covar_module_str'],
                        mean_module_str=lcls['mean_module_str'],
                        nonlinearity_hidden_m=lcls['nonlinearity_hidden_m'],
                        nonlinearity_hidden_k=lcls['nonlinearity_hidden_k'],
                        nonlinearity_output_m=lcls['nonlinearity_output_m'],
                        nonlinearity_output_k=lcls['nonlinearity_output_k'],
                        likelihood_str=lcls['likelihood_str'],      # only in ours #TODO
                        kernel_nn_layers=lcls['kernel_nn_layers'],
                        mean_nn_layers=lcls['mean_nn_layers'],
                        optimizer='Adam', lr=lcls['lr'],
                        lr_decay=lcls['lr_decay'],                  # only in ours #TODO
                        task_batch_size=lcls['task_batch_size'],    # only in ours
                        normalize_data=lcls['normalize_data'],
                        optimize_noise=lcls['optimize_noise'],
                        noise_std=lcls['noise_std'],
                        optimize_lengthscale=lcls['optimize_lengthscale'],
                        lengthscale_fix=lcls['lengthscale_fix'],
                        random_seed=random_seed,
                        logger=logger,                              # only in ours #TODO
                        svi_batch_size=10, cov_type='diag',         # only in ours VI
                        ts_data=lcls['ts_data']
                    )
            elif mode=='meta_fedavg':
                gp = GPRegressionMetaFedAvg(
                    clients_train_data, learning_mode='both', min_noise_std=1e-3,
                    num_iter_fit=lcls['num_iter_fit'], feature_dim=lcls['feature_dim'],
                    covar_module_str=lcls['covar_module_str'],
                    mean_module_str=lcls['mean_module_str'],
                    nonlinearity_hidden_m=lcls['nonlinearity_hidden_m'],
                    nonlinearity_hidden_k=lcls['nonlinearity_hidden_k'],
                    nonlinearity_output_m=lcls['nonlinearity_output_m'],
                    nonlinearity_output_k=lcls['nonlinearity_output_k'],
                    kernel_nn_layers=lcls['kernel_nn_layers'],
                    mean_nn_layers=lcls['mean_nn_layers'],
                    optimizer='Adam', lr=lcls['lr'],
                    lr_decay=lcls['lr_decay'],
                    optimize_noise=lcls['optimize_noise'], noise_std=lcls['noise_std'],
                    lengthscale_fix=lcls['lengthscale_fix'], optimize_lengthscale=lcls['optimize_lengthscale'],
                    random_seed=random_seed,
                    ts_data=lcls['ts_data'],
                    weight_decay=0.0, #TODO
                    logger=logger,
                    task_batch_size=lcls['task_batch_size'], normalize_data=lcls['normalize_data']
                )

            logger.info('params before training')
            logger.info(print_gp_prior_params(gp))

            # 2. fit model
            if mode=='ours':
                if approx_method=='SVGD':
                    cont_fit_margin = 4e-5
                else:
                    cont_fit_margin = None

                gp.meta_fit(
                    valid_tuples=clients_data,
                    log_period=log_period,
                    criteria=criteria,
                    early_stopping=lcls['early_stopping'],  # only in ours #TODO
                    cont_fit_margin=cont_fit_margin,        # only in ours #TODO
                    max_iter_fit=lcls['max_iter_fit']       # only in ours #TODO
                    )

            elif mode == 'meta_fedavg':
                gp.fit(clients_data,log_period=log_period)

            elif mode in ['personal', 'global']:
                gp.fit(
                    x_valid, y_valid,
                    log_period=log_period, verbose=verbose)

            logger.info('params after training')
            logger.info(print_gp_prior_params(gp))


            # 3. eval
            msg, valid_res, train_res = _eval(
                gp=gp, client_num_fix=client_num_fix,
                clients_data=clients_data, criteria=criteria)
            logger.info(msg)


            # 4. compare
            for criterion in criteria:
                if calc_mean(valid_res[criterion]) < calc_mean(best_res_all[criterion]['criterion_valid']):
                    best_res_all[criterion]['criterion_valid'] = copy.deepcopy(valid_res[criterion])
                    best_res_all[criterion]['criterion_train'] = copy.deepcopy(train_res[criterion])
                    best_res_all[criterion]['setup'] = copy.deepcopy(setup)
                    best_gp[criterion] = gp
                    if not pf_scheduler is None:
                        best_res_all[criterion]['setup'] += (prior_factor,)


            # 5. pf scheduler take a step
            if pf_scheduler is None:
                done = True
            else:
                pf_scheduler.step(
                        train_criterion=train_res[criteria[0]],
                        valid_criterion=valid_res[criteria[0]])
                done = pf_scheduler.done


        # progress info
        if not pf_scheduler is None:
            msg = '\n' + pf_scheduler.msg
        msg += '\n{:2.1f} percent completed.\n'.format((model_ind+1)/len(setups)*100)
        logger.info(msg)


    # ----- FINAL RESULTS -----
    if not client_num_fix is None:
        msg = '\n[RES] best for client {:2.0f}:'.format(client_num_fix)
    else:
        msg = '[RES] best over all:'
    for criterion in criteria:
        msg += '\nwith ' + criterion + 'criterion: train = {:2.4f}, valid = {:2.4f}'.format(
            calc_mean(best_res_all[criterion]['criterion_train']),
            calc_mean(best_res_all[criterion]['criterion_valid']))
        msg += '\nobtained by: '
        # convert best setup to dict
        if not (pf_scheduler is None or 'prior_factor' in setup_keys):
            setup_keys.append('prior_factor')
        best_setup_dict = dict.fromkeys(setup_keys)
        print(best_res_all[criterion])
        print(best_res_all[criterion]['setup'])
        for key, value in zip(setup_keys, list(best_res_all[criterion]['setup'])):
            if key.startswith('nonlinearity'):
                if not value is None:
                    value = value.__name__
            best_setup_dict[key] = value
            msg += '\n' + key + ': ' + str(value)
        best_res_all[criterion]['setup'] = best_setup_dict
    logger.info(msg)
    return best_gp, best_res_all



# ------------------------ PARSE INPUTS ------------------------
def _parse_inputs(options, mode, **kwargs):
    msg = ''
    # required input arguements
    req_inputs = ['optimize_noise', 'noise_std',
        'optimize_lengthscale', 'lengthscale_fix',
        'covar_module_str', 'mean_module_str',
        'mean_nn_layers', 'kernel_nn_layers',
        'nonlinearity_hidden_m', 'nonlinearity_hidden_k',
        'nonlinearity_output_m','nonlinearity_output_k',
        'feature_dim', 'lr', 'lr_decay', 'task_batch_size',
        'normalize_data', 'num_iter_fit', 'ts_data']
    if not mode=='meta_fedavg':
        req_inputs.extend([
            'max_iter_fit', 'early_stopping', 'num_particles',
            'bandwidth', 'hyper_prior_dict', 'n_threads'
        ])
    # add relevant keys in kwargs to options
    for key, value in kwargs.items():
        if key in req_inputs:
            if not key in options.keys():
                options[key] = value
            else: # key appears both in kwargs and options
                if not options[key] == value: # mismatch
                    msg += '[WARNING] mismatch in ' + key +'.\n'
        else:
            msg += '[WARNING] input argument ' + key + 'not used.\n'

    # all required input arguements must be in options
    missing_inputs = list(set(req_inputs).difference(options.keys()))
    if len(missing_inputs)>0:
        msg += '[ERROR] missing the following reuired inputs '.join([str(s) for s in missing_inputs])
        return msg, None, None, None

    # divide to fixed and variable options
    fix_options, msg_hsp, setups, setup_keys = handle_search_space(options)
    msg += msg_hsp

    # convert all common info to variables
    lcls = fix_options
    if 'nn_layers' in lcls.keys():
        lcls['mean_nn_layers'] = value
        lcls['kernel_nn_layers'] = value
    return msg, lcls, setups, setup_keys





# ------------------------ HANDLE SEARCH SPACE ------------------------
def handle_search_space(options):
    '''
    divids options to fixed and variable options and returns
    all setups for training
    '''
    fix_options = {}
    var_options = {}
    for key, value in options.items():
        if isinstance(value, list) or isinstance(value, np.ndarray):
            if len(value)==0: # empty []
                fix_options[key] = value
            elif len(value)==1: # not really a list!
                fix_options[key] = value[0]
            elif len(value)>1:
                var_options[key] = value
        else:
            fix_options[key] = value

    # --- log common info --- #
    msg = 'General model setup:\n'
    for key, value in fix_options.items():
        msg += key + ': ' + str(value) + ', '


    # create product of variable options
    setups = list(itertools.product(*var_options.values()))
    setup_keys = list(var_options.keys())
    ''' modifying the search space:
    1) remove NN size if none of kernel and mean are NN
    2) only use 1 SVGD BW if 1 particle '''
    for ind, s in enumerate(setups):
        # only use 1 SVGD BW if 1 particle
        if ('bandwidth' in setup_keys) and ('num_particles' in setup_keys) and (s[setup_keys.index('num_particles')]==1):
                z = list(s)
                z[setup_keys.index('bandwidth')] = 1#options['bandwidth'][0]
                setups[ind] =z
        # remove NN size if none of kernel and mean are NN
        if ('covar_module_str' in fix_options.keys()) and (fix_options['covar_module_str']=='NN'): continue
        if ('covar_module_str' in var_options.keys()) and (s[setup_keys.index('covar_module_str')]=='NN'): continue
        if ('mean_module_str' in fix_options.keys()) and (fix_options['mean_module_str']=='NN'): continue
        if ('mean_module_str' in var_options.keys()) and (s[setup_keys.index('mean_module_str')]=='NN'): continue
        # if reaches here, none of the mean and kernel where NN
        if 'nn_layers' in setup_keys:
            s[setup_keys.index('nn_layers')] = None # set NN layer size to None if not used
            setups[ind] = s # put back in setups


    # remove duplicates
    res = []
    [res.append(x) for x in setups if x not in res]
    setups = res

    return fix_options, msg, setups, setup_keys



# ------------------------ EVAL ------------------------
def _eval(gp, client_num_fix, clients_data, criteria):
    # if not gp.fitted:
    #     return '[WARN] GP not fitted', [], []


    # evaluate
    if client_num_fix is None: # evaluate on all clients
        # prepare train data
        num_clients = len(clients_data)
        clients_train_data = [None]*num_clients
        for n in np.arange(num_clients):
            x_obs, y_obs, _, _ = clients_data[n]
            clients_train_data[n] = (x_obs, y_obs, x_obs, y_obs)
        del n

        if isinstance(gp, GPRegressionMetaLearnedSVGD) and (not gp.best_particles is None):
            valid_res = dict.fromkeys(criteria)
            train_res = dict.fromkeys(criteria)
            # use best particles for each criterion
            for criterion in criteria:
                gp.particles = gp.best_particles[criterion]
                valid_res[criterion] = gp.eval_datasets(clients_data, get_full_list=True)[criterion]
                train_res[criterion] = gp.eval_datasets(clients_train_data, get_full_list=True)[criterion]
        elif isinstance(gp, GPRegressionMetaLearnedVI) and (not gp.best_posterior is None):
            valid_res = dict.fromkeys(criteria)
            train_res = dict.fromkeys(criteria)
            # use best particles for each criterion
            for criterion in criteria:
                gp.posterior = gp.best_posterior[criterion]
                valid_res[criterion] = gp.eval_datasets(clients_data, get_full_list=True)[criterion]
                train_res[criterion] = gp.eval_datasets(clients_train_data, get_full_list=True)[criterion]
        else:
            valid_res = dict.fromkeys(criteria)
            train_res = dict.fromkeys(criteria)
            # use best particles for each criterion
            for criterion in criteria:
                valid_res = gp.eval_datasets(clients_data, get_full_list=True)
                train_res = gp.eval_datasets(clients_train_data, get_full_list=True)

        if len(clients_data)==1:
            for key in valid_res.keys():
                if isinstance(valid_res[key], list):
                    valid_res[key] = valid_res[key][0]
                if isinstance(train_res[key], list):
                    train_res[key] = train_res[key][0]
    else:
        x_train, y_train, x_valid, y_valid = clients_data[client_num_fix]
        valid_res = dict.fromkeys(['nll', 'rmse', 'calibr', 'rsmse'], None)
        train_res = dict.fromkeys(['nll', 'rmse', 'calibr', 'rsmse'], None)
        if isinstance(gp, GPRegressionLearned):
            valid_res['nll'], valid_res['rmse'], valid_res['calibr'] = gp.eval(
                x_valid, y_valid)
            train_res['nll'], train_res['rmse'], train_res['calibr'] = gp.eval(
                x_train, y_train)
        else:
            valid_res['nll'], valid_res['rmse'], valid_res['calibr'] = gp.eval(
                x_train, y_train, x_valid, y_valid)
            train_res['nll'], train_res['rmse'], train_res['calibr'] = gp.eval(
                x_train, y_train, x_train, y_train)
        valid_res['rsmse'] = [valid_res['rmse']/np.std(y_valid.flatten())]
        train_res['rsmse'] = [train_res['rmse']/np.std(y_train.flatten())]


    # # convert cur_res[key] from tuple to list
    # for key in cur_res.keys():
    #     cur_res[key] = list(cur_res[key])
    msg=''
    for criterion in criteria:
        msg += '\nTrain-' + criterion + ': {:2.4f}, Valid-'.format(
            calc_mean(train_res[criterion]))
        msg +=  criterion + ': {:2.4f}'.format(
            calc_mean(valid_res[criterion]))

    return msg, valid_res, train_res



# ------------------------