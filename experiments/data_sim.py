import os, sys, copy, torch
import torch.nn as nn
import numpy as np
from scipy.stats import truncnorm


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, PROJECT_DIR)
import config


class ClientsEnv():

    def __init__(self, random_state=None):
        if random_state == None:
            self.random_state = np.random
        else:
            self.random_state = random_state
        self.x_mean, self.y_mean = None, None
        self.x_std, self.y_std = None, None


    def generate_meta_train_data(self, n_tasks: int, n_samples: int) -> list:
        raise NotImplementedError


    def generate_meta_test_data(self, n_tasks:int, n_samples_context: int, n_samples_test: int) -> list:
        raise NotImplementedError


from numpy import matlib as mb
class MVN:
    def __init__(self, mean, cov):
        self.ndim = cov.shape[0]
        self.mean = mean.reshape(self.ndim)
        self.cov = cov.reshape(self.ndim, self.ndim)
        self.inv_cov = np.linalg.inv(self.cov)
        self.det_cov = np.linalg.det(self.cov)


    def dlnprob(self, thetas):
        '''
        thetas in R^{t*n_dim} is a set of t weight vectors
        returns a vector of length t, where the ith element is
        d/d theta_i ln pr(theta_i | mean_weights, cov_weights)
        '''
        assert isinstance(thetas, np.ndarray)
        assert thetas.shape[1] == self.ndim
        # assert thetas[:, 0].flatten() == np.ones(self.ndim, 1).flatten()
        return -1*np.matmul(thetas-mb.repmat(self.mean, thetas.shape[0], 1), self.inv_cov)

    def pdf(self, theta):
        return self.forward(theta.reshape(1, -1))

    def forward(self, thetas):
        '''
        thetas in R^{t*n_dim} is a set of t weight vectors
        returns multivariate Gaussian pdf on array thetas (in R^t)
        '''
        N = np.sqrt((2*np.pi)**self.ndim * self.det_cov)
        # einsum calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', thetas-self.mean, self.inv_cov, thetas-self.mean)
        return np.exp(-fac / 2) / N

    def sample(self, num_samples, random_state):
        return random_state.multivariate_normal(
            mean=self.mean, cov=self.cov, size=num_samples)


class GMM:
    def __init__(self, means, covs, weights):
        self.num_modes = len(means)
        assert len(weights) == self.num_modes
        assert len(covs) == self.num_modes
        assert sum(weights) == 1
        self.dist = [None] * self.num_modes
        for ind in np.arange(self.num_modes):
            self.dist[ind] = MVN(means[ind], covs[ind])
        self.weights = weights


    def dlnprob(self, theta):
        prob = self.forward(theta) + 1e-6 # TO avoid division by 0
        dprob = 0
        for weight, dist in zip(self.weights, self.dist):
            '''
            dist.dlnprob(theta) \in \R^{num_particles, num_dimensions}, where element i,j
            is the derivative of ln(prob(particle i)) w.r.t element j of particle i.
            dist.forward(theta) \in \R^{num_particles} is the probability for each particle
            np.multiply performs elementwise multiplication => row i of dist.dlnprob(theta)
            is multiplied by row i of dist.forward(theta)
            '''
            dprob += weight * np.multiply(dist.dlnprob(theta), dist.forward(theta).reshape(-1,1))
            # print('dln', dist.dlnprob(theta).shape)
            # print('frw', dist.forward(theta).reshape(-1,1).shape)
            # print('dprob', dprob.shape)
        #dprob = sum([weight * np.multiply(dist.dlnprob(theta).flatten(), dist.forward(theta).flatten()) for weight, dist in zip(self.weights, self.dist)])
        # print('res', np.divide(dprob, prob).shape)
        return np.divide(dprob, prob.reshape(-1,1))

    def forward(self, pos):
        """Return the Gaussian mixture distribution on array pos."""
        res = 0
        for weight, dist in zip(self.weights, self.dist):
            res += weight * dist.forward(pos)
        return res

    def sample(self, num_samples, random_state):
        self.num_samples_per_mode = [int(self.weights[i]*num_samples) for i in np.arange(self.num_modes)]
        remained_num = num_samples-sum(self.num_samples_per_mode)
        for ind in np.arange(remained_num):
            self.num_samples_per_mode[ind] += 1
        samples = None
        for mode_num in np.arange(self.num_modes):
            new_samples = self.dist[mode_num].sample(self.num_samples_per_mode[mode_num], random_state)
            samples = new_samples if samples is None else np.concatenate((samples, new_samples), axis=0)
        return samples



# --------------------------------------------
class PVDataset(ClientsEnv):

    sys.path.insert(1, config.SYNTH_PV_DIR)
    from city_pv_multi_modal import CityPV_MultiModal

    def __init__(self, env_options, random_state=None):
        super().__init__(random_state)

        # check provided options
        req_fields = [# required for building the task env
                'city_names', 'tilt_std', 'az_std', 'weather_dev',
                'irrad_std' ,'altitude_dev', 'shadow_peak_red',
                # required for simulating pv
                'module_name', 'inverter_name'
                # required for generating data
                'train_scenarios']
        optional_fields = [# optional for building the task env
                'tilt_mean', 'az_mean',
                # optional for simulating pv data
                'lags', 'hours', 'months',
                'num_clients', 'num_clients_per_mode',
                'use_station_irrad_direct', 'use_station_irrad_diffuse', 'delay_irrad',
                # optional for generating datasets
                'remove_constant_cols'
                ]

        assert [x in env_options.keys() for x in req_fields]
        assert [x in req_fields + optional_fields for x in env_options.keys()]
        assert 'num_clients' in env_options.keys() or 'num_clients_per_mode' in env_options.keys()

        # parse optional options
        for key in optional_fields:
            if not key in env_options.keys():
                if key in ['use_station_irrad_direct', 'use_station_irrad_diffuse', 'delay_irrad', 'remove_constant_cols']:
                    env_options[key] = True
                else:
                    env_options[key] = None
        if env_options['num_clients_per_mode'] is not None:
            if env_options['num_clients'] is None:
                env_options['num_clients'] = np.sum(env_options['num_clients_per_mode'])
            else:
                assert env_options['num_clients'] == np.sum(env_options['num_clients_per_mode'])
        if env_options['num_clients'] is not None and env_options['num_clients_per_mode'] is None:
            env_options['num_clients_per_mode'] = [int(env_options['num_clients']/len(env_options['city_names']))]*len(env_options['city_names'])
            env_options['num_clients'] = np.sum(env_options['num_clients_per_mode'])

        # create task environment
        self.task_environment = CityPV_MultiModal(city_names=env_options['city_names'],
                tilt_mean=env_options['tilt_mean'], az_mean=env_options['az_mean'],
                tilt_std=env_options['tilt_std'], az_std=env_options['az_std'],
                weather_dev=env_options['weather_dev'], irrad_std=env_options['irrad_std'],
                altitude_dev=env_options['altitude_dev'], shadow_peak_red=env_options['shadow_peak_red'],
                random_state=self.random_state)

        # summarize info
        env_options['info'] = '{:2.0f} households at '.format(env_options['num_clients'])+ " ".join(env_options['city_names']) + ' - '
        for key in ['tilt_std', 'az_std', 'weather_dev', 'irrad_std', 'altitude_dev', 'shadow_peak_red']:
            if not isinstance(env_options[key], list): # TODO
                env_options['info'] += key+': {:.1f}, '.format(env_options[key])
        for name_str in ['module_name', 'inverter_name']:
            if len(env_options[name_str])==1:
                env_options['info'] += 'same ' + name_str + ', '
            elif len(env_options[name_str])==env_options['num_clients']:
                env_options['info'] += 'different ' + name_str + ', '
            else:
                env_options['info'] += + name_str + 'not specified, '

        self.env_dict = env_options
        self._simulate_pv()


    def _simulate_pv(self):
        self.task_environment.simulate_pv(num_clients_per_mode=self.env_dict['num_clients_per_mode'],
                                          module_name=self.env_dict['module_name'], inverter_name=self.env_dict['inverter_name'],
                                          lags=self.env_dict['lags'], months=self.env_dict['months'],
                                          hours=self.env_dict['hours'],
                                          use_station_irrad_direct=self.env_dict['use_station_irrad_direct'],
                                          use_station_irrad_diffuse=self.env_dict['use_station_irrad_diffuse'],
                                          delay_irrad=self.env_dict['delay_irrad'])
        # properties of task env that might have changed
        self.env_dict['months'] = self.task_environment.months
        self.env_dict['hours'] = self.task_environment.hours
        self.env_dict['lags'] = self.task_environment.lags
        # new properties
        self.env_dict['clients_config']=self.task_environment.clients_config


    def generate_clients_data(self, shuffle=False):
        self.env_dict['feature_names'] = None
        for scenario_name, scenario in self.env_dict['train_scenarios'].items():
            self.task_environment.construct_regression_matrices(
                m_train=scenario['m_train'], train_years=scenario['train_years'],
                valid_years=scenario['valid_years'], shuffle=shuffle,
                remove_constant_cols=self.env_dict['remove_constant_cols'])
            self.env_dict['train_scenarios'][scenario_name]['clients_data'] = copy.deepcopy(self.task_environment.clients_data_tuple)
            self.env_dict['train_scenarios'][scenario_name]['time_series'] = copy.deepcopy(self.task_environment.clients_time_series)

            if self.env_dict['feature_names'] is None:
                self.env_dict['feature_names'] = self.task_environment.feature_names
            else:
                assert self.env_dict['feature_names'] == self.task_environment.feature_names
        return self.env_dict



def remove_feature(env_dict, feature_name, in_place=False):
    '''
    removes a specified feature from task environemnt.
    inplace operation
    '''
    if not in_place:
        env_dict = copy.deepcopy(env_dict)
    assert isinstance(feature_name, str)
    assert feature_name in env_dict['feature_names']
    # find index of features except this feature
    feature_index_to_keep = np.delete(
                                np.arange(len(env_dict['feature_names'])),
                                env_dict['feature_names'].index(feature_name)
                                )
    print('feature to keep ', *feature_index_to_keep)
    # remove from feature_names
    env_dict['feature_names'].remove(feature_name)
    # no need to remove from time_series
    # remove from X_train and X_test at each scenario and for all clients
    for client_num in np.arange(env_dict['num_clients']):
        for scenario in env_dict['train_scenarios']:
            x_train, y_train, x_valid, y_valid = env_dict['train_scenarios'][scenario]['clients_data'][client_num]
            x_train = x_train[:, feature_index_to_keep]
            x_valid = x_valid[:, feature_index_to_keep]
            env_dict['train_scenarios'][scenario]['clients_data'][client_num] = (x_train, y_train, x_valid, y_valid)
            assert x_train.shape[1] == len(env_dict['feature_names'])
            assert x_valid.shape[1] == len(env_dict['feature_names'])
    if in_place:
        return
    else:
        return env_dict



# -------------------------------------------------------------------
class ToyDataset(ClientsEnv):

    def __init__(self, noise_std=0.05, length_scales=[0.3, 3], random_state=None, mean_type='poly'):
        assert len(length_scales)==2
        assert mean_type in ['poly', 'nn', 'Cauchy']
        self.noise_std = noise_std
        self.length_scales = length_scales
        self.mean_type = mean_type
        self.random_state = random_state
        self.x_low, self.x_high = -0.8, 0.8

        # 2 nn means
        if self.mean_type =='nn':
            self.mean_nns = [None]*2
            for mode_num in np.arange(2):
                self.mean_nns[mode_num] = nn.Sequential(
                    nn.Linear(1, 8), nn.Tanh(),
                    nn.Linear(8, 8), nn.Tanh(),
                    nn.Linear(8, 1)
                )
            # for i in [0, 2]:
            #     self.mean_nn1[i].weight.data = 0.4*torch.ones(self.mean_nn1[i].weight.data.shape)
        elif self.mean_type=='Cauchy':
            self.cauchy_locs = [-1, 2]
            self.cauchy_weights = [6,1]
            self.cuachy_consts = [3, -2]
        elif self.mean_type=='poly':
            self.poly_roots = [
                np.array([ 2.13090278,  1.04920788,  0.77719797, -0.79079416,  0.47808944, 0.41619282,  0.63380681]),
                np.array([-0.93890047, -0.53067272, -0.83405262, -1.41970533,  0.11923567, 0.11692263,  0.69670114])]
            # [
            #     self.random_state.normal(loc=0.7, scale=0.8, size=7),
            #     self.random_state.normal(loc=-0.5, scale=0.7, size=7)
            # ]
            self.poly_coeffs = [
                1, -1
            ]



    def generate_clients_data(self, num_clients, n_train, n_test, weight_modes):
        assert n_train > 0
        assert len(weight_modes)==2 and sum(weight_modes)==1
        noisy_data_tups = []
        noise_free_data_tups = []

        for i in range(num_clients):
            mode_num = 0 if i<int(num_clients*weight_modes[0]) else 1
            X = truncnorm.rvs(self.x_low, self.x_high, loc=0, scale=1,
                              size=(n_train + n_test, 1), # 1D
                              random_state=self.random_state)
            Y, f = self._gp_fun_from_prior(X, mode_num)
            noisy_data_tups.append(
                (X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]))
            noise_free_data_tups.append(
                (X[:n_train], f[:n_train], X[n_train:], f[n_train:]))

        return noisy_data_tups, noise_free_data_tups

    def _mean(self, x, mode_num):
        if self.mean_type=='nn':
            gp_mean = self.mean_nns[mode_num].forward(torch.from_numpy(np.float32(x))).detach().numpy()
        elif self.mean_type=='cauchy':
            gp_mean = self.cauchy_weights[mode_num] / (
                np.pi * (1 + (np.linalg.norm(x - self.cauchy_locs[mode_num]* np.ones(x.shape[-1]), axis=-1))**2)
                ) + self.cuachy_consts[mode_num]
        elif self.mean_type=='poly':
            gp_mean=np.ones((x.shape[0], 1))
            for deg in np.arange(7):
                gp_mean = np.multiply(gp_mean, (x-self.poly_roots[mode_num][deg]*np.ones(gp_mean.shape)))
            gp_mean = np.multiply(gp_mean, self.poly_coeffs[mode_num])
        assert gp_mean.shape[0]==x.shape[0] and gp_mean.shape[1]==1
        return gp_mean.reshape(-1,1)

    def _gp_fun_from_prior(self, X, mode_num):
        n = X.shape[0]
        def kernel(a, b, length_scale):
            sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
            return np.exp(-.5 * (1 / length_scale**2) * sqdist)

        K_ss = kernel(X, X, self.length_scales[mode_num])
        L = np.linalg.cholesky(K_ss + 1e-8 * np.eye(n))
        f = self._mean(X, mode_num) + np.dot(L, self.random_state.normal(scale=0.2, size=(n, 1))).reshape(-1,1)
        assert f.shape[0]==n and f.shape[1]==1
        y = f + self.random_state.normal(scale=self.noise_std, size=f.shape)
        return y.reshape(-1, 1), f.reshape(-1, 1)



# -----------------------------------------------------

if __name__ == "__main__":

    random_state = 3

    # geographical characteristics of the location
    latitude=46.520
    longitude=6.632
    city_name='Lausanne'
    altitude=496
    timezone='Etc/GMT-1'

    # clients distribution
    num_modes = 1
    weight_modes = [1/num_modes] * num_modes
    mean_tilt  = latitude
    mean_azimuth = 180
    sigma_tilt = 15
    sigma_azimuth  = 45
    mean_weights = [[mean_tilt, mean_azimuth]]
    cov_weights = [np.diag([sigma_tilt**2, sigma_azimuth **2])]

    # FL info
    num_clients = 25
    num_clients_per_mode = [int(weight_modes[i]*num_clients) for i in np.arange(num_modes)]*num_modes


    # generate data from each mode
    task_environment = PVDataset(mean_weights=mean_weights, cov_weights=cov_weights,
                                    city_name=city_name,
                                    random_state=random_state)


    print('[INFO] generating data for {:2.0f} clients'.format(num_clients))
    clients_data, clients_train_ts, clients_test_ts = task_environment.generate_clients_data(num_clients=num_clients,
                                                          weight_modes=weight_modes)
    #print(task_environment.true_data_dist)
    #print(y_obs.shape)



