
def visualize_env(env_dict, num_days=5, year=2018, scenario_name=None):
    ''' visualizes data from the given year '''
    # get data tuples and tie series corresponding to the given scenario
    if scenario_name is None:
        scenario_name = list(env_dict['train_scenarios'].keys())[0]
    clients_data = env_dict['train_scenarios'][scenario_name]['clients_data']
    time_series = env_dict['train_scenarios'][scenario_name]['time_series']

    labels = [int(x) for x in env_dict['clients_config']['azimuth']]
    if len(env_dict['city_names'])==1:
        colors = ["#%06x" % np.random.randint(0, 0xFFFFFF) for _ in range(env_dict['num_clients'])]
    else:
        colors =  [(0,1,0), (0,0,1), (1,0,1)]#list(itertools.product((0,1), (0,1), (0,1)))[1:-1]
        #np.random.shuffle(colors)

    # plot num_days days in the given year
    for month in env_dict['months']:
        client_num = 0
        _, axs = plt.subplots(2, 1, figsize = (30, 8*2))
        for mode_num in np.arange(len(env_dict['city_names'])):
            for i in np.arange(env_dict['num_clients_per_mode'][mode_num]):
                # choose color
                if len(env_dict['city_names'])==1:
                    color = colors[client_num]
                else:
                    intensity = 0.2+(1-0.2)*i/(env_dict['num_clients_per_mode'][mode_num]-1) if env_dict['num_clients_per_mode'][mode_num]>1 else 0.5
                    color = tuple(x * intensity for x in colors[mode_num])

                _, y_train, _, y_valid = clients_data[client_num]
                y_mean, y_std = np.mean(y_train, axis=0), np.std(y_train, axis=0) + 1e-8

                # normalized and unnormalized data
                ts_unnorm = time_series[client_num]
                ts_unnorm = ts_unnorm.loc[(ts_unnorm['month']==month) & (ts_unnorm['year']==year), :]
                ts_unnorm = ts_unnorm.loc[ts_unnorm['hour_day'].isin(env_dict['hours']), :]
                ts_unnorm = ts_unnorm.iloc[np.arange(num_days*len(env_dict['hours'])), :]
                ts_norm = (ts_unnorm.loc[:, 'target'] - y_mean) / y_std

                #if mode_num>-1:
                for day_num in np.arange(num_days):
                    inds_day = day_num*len(env_dict['hours']) + np.arange(len(env_dict['hours']))
                    if day_num==0:
                        axs[0].plot(inds_day, ts_unnorm.loc[:, 'target'].iloc[inds_day],
                                    label=labels[client_num], lw=1, c=color)
                        axs[1].plot(inds_day, ts_norm.iloc[inds_day],
                                    label=labels[client_num], lw=1, c=color)
                    else:
                        axs[0].plot(inds_day, ts_unnorm.loc[:, 'target'].iloc[inds_day],
                                    lw=1, c=color)
                        axs[1].plot(inds_day, ts_norm.iloc[inds_day],
                                    lw=1, c=color)


                client_num += 1
        axs[0].set_ylabel('power unnormalized')
        axs[1].set_ylabel('power normalized')
        for ax in axs:
            ax.set_title(env_dict['info'] + ' - month ' + str(month))
            for i in np.arange(num_days+1):
                ax.axvline(x=len(env_dict['hours'])*i, ymin=0, ymax=1)
            tick_locs = np.linspace(0, num_days*len(env_dict['hours']),
                                    num=int(num_days*len(env_dict['hours'])/4),
                                    endpoint=False).astype(int)
            ax.set_xticks(tick_locs, labels=ts_unnorm.loc[:, 'hour_day'].iloc[tick_locs])
            ax.set_xlabel(str(num_days) + ' days of data')
            ax.legend()
        plt.show()
