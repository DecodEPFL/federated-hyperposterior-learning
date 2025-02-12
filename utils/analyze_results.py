import sys, os, random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.abstract import _handle_input_dimensionality


 # --------- PLOT PREDICTIONS ---------
def plot_preds_with_res(
        env_dict, env_dict_res, scenario_name, clients_subset, criteria,
         models_all, use_res=None, methods=None, cols_subset=None, num_days=7, plot_mode = 'valid'):
    '''
    plot predictions of trained models for different methods
    '''
    scenario_name_plt = scenario_name

    clients_data = env_dict['train_scenarios'][scenario_name_plt]['clients_data']
    if not env_dict_res is None:
        res_data = env_dict_res['train_scenarios'][scenario_name_plt]['clients_data']
    else:
        res_data = env_dict['train_scenarios'][scenario_name_plt]['clients_data']

    _, axs = plt.subplots(len(clients_subset),2,
                            figsize=(24,6*len(clients_subset)))

    if methods is None:
        methods = models_all.keys()
    if use_res is None:
        use_res = [False]*len(methods)
    else:
        assert len(use_res)==len(methods)

    plot_true = True
    for method_ind, method in enumerate(methods):

        if models_all[method] is None:
            continue
        model_tmp = models_all[method][scenario_name]
        if model_tmp is None:
            continue

        # ----- FEATURES SUBSET -----
        cols_subset = np.arange(clients_data[0][0].shape[1]) if cols_subset is None else cols_subset
        # if 'features_subset' in options[method].keys():
        #     # selecting a subset of features given by their names
        #     if len(options[method]['features_subset'])<len(env_dict['feature_names']):
        #         cols_subset = [i for i,x in enumerate(env_dict['feature_names']) if x in options[method]['features_subset']]


        for client_ind, client_num in enumerate(clients_subset):
            # select model
            if isinstance(model_tmp[criteria], list):      # one model per client
                model = model_tmp[criteria][client_num]
            else:     # one model for all clients
                model = model_tmp[criteria]

            # get data
            _, y_train, _, y_valid = clients_data[client_num]
            if use_res[method_ind]:
                x_res_train, y_res_train, x_res_valid, y_res_valid = res_data[client_num]
            else:
                x_res_train, y_res_train, x_res_valid, y_res_valid = clients_data[client_num]
            x_res_train = x_res_train[:, cols_subset]
            x_res_valid = x_res_valid[:, cols_subset]
            x_res_train, y_res_train = _handle_input_dimensionality(x_res_train, y_res_train)
            x_res_valid, y_res_valid = _handle_input_dimensionality(x_res_valid, y_res_valid)

            if plot_mode=='valid':
                pred_mean, pred_std = model.predict(x_res_train, y_res_train, x_res_valid)
                pred_mean = pred_mean.flatten() - y_res_valid.flatten() + y_valid.flatten()
                y_true = y_valid.flatten()
            elif plot_mode=='train':
                pred_mean, pred_std = model.predict(x_res_train, y_res_train, x_res_train)
                pred_mean = pred_mean.flatten() - y_res_train.flatten() + y_train.flatten()
                y_true=y_train.flatten()

            for month_ind, month in enumerate(env_dict['months']):
                if len(clients_subset)==1:
                    ax = axs[month_ind]
                else:
                    ax = axs[client_ind][month_ind]
                inds = np.arange(
                    month_ind*num_days*len(env_dict['hours']),
                    (month_ind+1)*num_days*len(env_dict['hours']))
                ax.plot(inds, pred_mean[inds], label=method)
                ax.fill_between(
                    inds,
                    pred_mean[inds]-1.96*pred_std[inds],pred_mean[inds]+1.96*pred_std[inds], alpha=0.3)
                if plot_true:
                    ax.plot(inds, y_true[inds], 'black', label='true')
                if plot_mode=='valid':
                    ax.set_title('predictions on validation set for client {:2.0f} - month {:2.0f}'.format(client_num, month))
                elif plot_mode=='train':
                    ax.set_title('predictions on training set for client {:2.0f} - month {:2.0f}'.format(client_num, month))
                ax.legend()
        plot_true = False

    plt.show()



def plot_preds(
        env_dict, scenario_name, clients_subset, criteria,
        methods, models_all, mode, options, num_days=7, plot_mode = 'valid'):
    '''
    plot predictions of trained models for different methods
    '''
    scenario_name_plt = scenario_name

    clients_data = env_dict['train_scenarios'][scenario_name_plt]['clients_data']

    _, axs = plt.subplots(len(clients_subset),1,
                            figsize=(48,6*len(clients_subset)))

    plot_true = True
    for method in methods:
        if models_all[method] is None:
            continue
        model_tmp = models_all[method][scenario_name]
        if model_tmp is None:
            continue

        # ----- FEATURES SUBSET -----
        cols_subset = np.arange(clients_data[0][0].shape[1])
        if 'features_subset' in options[method].keys():
            # selecting a subset of features given by their names
            if len(options[method]['features_subset'])<len(env_dict['feature_names']):
                cols_subset = [i for i,x in enumerate(env_dict['feature_names']) if x in options[method]['features_subset']]


        for client_ind, client_num in enumerate(clients_subset):
            # select mode
            if mode == 'personal':      # one model per client
                model = model_tmp[criteria][client_num]
            if mode == 'ours':      # one model for all clients
                model = model_tmp[criteria]

            # get data
            x_train, y_train, x_valid, y_valid = clients_data[client_num]
            x_train = x_train[:, cols_subset]
            x_valid = x_valid[:, cols_subset]
            x_train, y_train = _handle_input_dimensionality(x_train, y_train)
            x_valid, y_valid = _handle_input_dimensionality(x_valid, y_valid)

            if plot_mode=='valid':
                pred_mean, pred_std = model.predict(x_train, y_train, x_valid)
                y_true = y_valid
            elif plot_mode=='train':
                pred_mean, pred_std = model.predict(x_train, y_train, x_train)
                y_true=y_train

            if len(clients_subset)==1:
                ax = axs
            else:
                ax = axs[client_ind]
            inds = np.arange(pred_mean.shape[0])
            ax.plot(inds, pred_mean, label=method)
            print(inds.shape, pred_mean.shape, pred_std.shape)
            ax.fill_between(
                inds,
                pred_mean-1.96*pred_std,pred_mean+1.96*pred_std, alpha=0.3)
            if plot_true:
                ax.plot(inds, y_true, 'black', label='true')
            ax.set_title(
                'predictions on validation set for client {:2.0f} - months {:2.0f} and {:2.0f}'.format(
                    client_num, env_dict['months'][0], env_dict['months'][1]))
            ax.legend()
        plot_true = False
    del clients_data
    plt.show()



# ----------       BOX PLOTS      ----------
def perf_box_plots(
    env_dict, criterion, methods, results_all,
    clients_subset=None, plot_title=None, figsize=(10, 5), num_hyper_params=None):

    # sort methods by num hyper_params
    if not num_hyper_params is None:
        assert len(num_hyper_params.keys())==len(methods)
        num_hyper_params = dict(sorted(num_hyper_params.items(), key=lambda item: item[1]))
        methods=num_hyper_params.keys()

    num_clients = env_dict['num_clients']
    if clients_subset is None:
        clients_subset = np.arange(num_clients)
    num_rows = len(clients_subset) * len(env_dict['train_scenarios']) * len(methods)
    res_df = {'client_num': np.zeros(num_rows), 'scenario_name': [None]*num_rows,
            'valid_'+criterion: np.zeros(num_rows),
            'train_'+criterion: np.zeros(num_rows),
            'method':[None]*num_rows}
    n=0
    for method in methods:
        for scenario_name in env_dict['train_scenarios']:
            if results_all[method] is None:
                continue
            if not scenario_name in results_all[method].keys():
                continue
            if results_all[method][scenario_name] is None:
                continue
            for client_num in clients_subset:
                res_df['client_num'][n] = client_num
                res_df['scenario_name'][n] = scenario_name
                res_df['method'][n] = method if num_hyper_params is None else method + '(' + str(num_hyper_params[method]) + ')'
                r = results_all[method][scenario_name][criterion]
                if isinstance(r, list):
                    assert len(r) == num_clients
                    res_df['valid_'+criterion][n] = r[client_num]['criterion_valid']
                    res_df['train_'+criterion][n] = r[client_num]['criterion_train']
                elif isinstance(r, dict):
                    assert 'criterion_valid' in r.keys()
                    res_df['valid_'+criterion][n] = r['criterion_valid'][client_num]
                    res_df['train_'+criterion][n] = r['criterion_train'][client_num]
                else:
                    raise NotImplementedError
                n +=1

    res_df = pd.DataFrame(data=res_df)

    # plot
    plot_title = 'Model Performance' if plot_title==None else plot_title
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=figsize)
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
    random.seed(3)
    colors = random.sample(sorted(mcolors.CSS4_COLORS), len(methods))
    for ax, train_or_valid in zip(axs, ['train', 'valid']):
        # Add a horizontal grid to the plot, but make it very light in color
        # so we can use it for reading data values but not be distracting
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                    alpha=0.5)
        ax.set(
            axisbelow=True,  # Hide the grid behind plot objects
            title='comparing models ' + criterion + ' on ' + train_or_valid + ' set',
            xlabel='method',
            ylabel=train_or_valid+criterion,
        )

        sns.boxplot(y=train_or_valid+'_'+criterion, x='scenario_name',
                    data=res_df,
                    palette=colors,
                    hue='method',
                    ax=ax)
        if train_or_valid=='valid':
            plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        else:
            ax.legend([],[], frameon=False)

    fig.suptitle(plot_title)
    plt.tight_layout()
    plt.show()

    # print stats
    stats_df = res_df.groupby(['method', 'scenario_name']).agg(
        {'valid_'+criterion: ['mean', 'min', 'max', 'median']})
    stats_df.columns = ['valid_'+criterion + '_' + x for x in ['mean', 'min', 'max', 'median']]
    stats_df = stats_df.reset_index()
    stats_df.style.highlight_min(color='lightgreen', axis=0)
    return stats_df




# ----------       COMPUTE RESIDUALS CORRELATION MATRIX      ----------
def compute_res_corr(env_dict_res, scenario_name):
    fig, axs = plt.subplots(1, 2, figsize=(2*5, env_dict_res['num_clients']/2))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    corr_mat = {'train':np.zeros((env_dict_res['num_clients']+2, len(env_dict_res['feature_names']))),
                'valid':np.zeros((env_dict_res['num_clients']+2, len(env_dict_res['feature_names'])))}
    clients_residuals = env_dict_res['train_scenarios'][scenario_name]['clients_data']
    for j, train_or_valid in enumerate(['train', 'valid']):
        for i, client_num in enumerate(np.arange(env_dict_res['num_clients'])):
            x_train, res_train, x_valid, res_valid = clients_residuals[client_num]
            if train_or_valid =='train':
                df = pd.DataFrame(x_train, columns=env_dict_res['feature_names'])
                df['residuals'] = res_train.flatten()
            else:
                df = pd.DataFrame(x_valid, columns=env_dict_res['feature_names'])
                df['residuals'] = res_valid.flatten()

            corr_dict = df.corr().filter(['residuals']).drop(['residuals'])
            corr_mat[train_or_valid][i, :] = corr_dict.values.flatten()
        # put average over houses in the last row
        corr_mat[train_or_valid][-2, :] = np.mean(corr_mat[train_or_valid][:-2, :], axis=0)
        corr_mat[train_or_valid][-1, :] = np.mean(np.abs(corr_mat[train_or_valid][:-2, :]), axis=0)
        ax_ind=0
        sns.heatmap(np.abs(corr_mat[train_or_valid]), ax=axs[j],
                    cbar=ax_ind == 0, square=True,
                    cbar_ax=cbar_ax,
                    xticklabels=env_dict_res['feature_names'],
                    yticklabels=['house ' + str(x) for x in np.arange(env_dict_res['num_clients'])]+['av', 'av abs'])
                    #mask=np.zeros_like(corr_mat, dtype=bool),
                    #cmap=sns.diverging_palette(220, 10, as_cmap=True),
        axs[j].set_title('correlation of residuals with features - ' + train_or_valid)
        axs[j].set_xlabel('features')
        axs[j].set_ylabel('houses')
    plt.show()
    return corr_mat

