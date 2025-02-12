import torch, sys, os
import numpy as np

sys.path.append(os.path.realpath(os.path.dirname(__file__))+'/../')

from server.models import AffineTransformedDistribution, EqualWeightedMixtureDist
from server.hyper_posterior import HyperPosterior
from server.util import _handle_input_dimensionality
from server.abstract import RegressionModelMetaLearned
from server.GPR_meta_svgd import GPRegressionMetaLearnedSVGD
from config import device


class TrainedSVGD(GPRegressionMetaLearnedSVGD, RegressionModelMetaLearned):

    def __init__(self, serialized_mdl, ts_data):

        """

        """
        if not 'random_seed' in serialized_mdl.keys():
            serialized_mdl['random_seed'] = 3
        RegressionModelMetaLearned.__init__(self, serialized_mdl['normalize_data'], serialized_mdl['random_seed'])
        self.ts_data = ts_data
        if not ('optimize_lengthscale' in serialized_mdl.keys() and 'lengthscale_fix' in serialized_mdl.keys()):
            serialized_mdl['optimize_lengthscale'] = True
            serialized_mdl['lengthscale_fix'] = None
        if not 'likelihood_str' in serialized_mdl.keys():
            print('[WARN] likelihood_str not found. using default (Gaussian)')
            serialized_mdl['likelihood_str'] = 'Gaussian'
        # set the same nonlinearity for mean and kernel if only one is given
        if 'nonlinearity_hidden' in serialized_mdl.keys():
            serialized_mdl['nonlinearity_hidden_m'] = serialized_mdl['nonlinearity_hidden']
            serialized_mdl['nonlinearity_hidden_k'] = serialized_mdl['nonlinearity_hidden']
            del serialized_mdl['nonlinearity_hidden']
        if 'nonlinearity_output' in serialized_mdl.keys():
            serialized_mdl['nonlinearity_output_m'] = serialized_mdl['nonlinearity_output']
            serialized_mdl['nonlinearity_output_k'] = serialized_mdl['nonlinearity_output']
            del serialized_mdl['nonlinearity_output']


        for key, value in serialized_mdl.items():
            # set GPR_meta_SVGD attributes
            nn_keys = ['nonlinearity_hidden_m', 'nonlinearity_output_m',
                       'nonlinearity_hidden_k', 'nonlinearity_output_k',
                       'mean_module_str', 'covar_module_str']
            if not key in nn_keys:
                if isinstance(value, torch.Tensor):
                    setattr(self, key, value.to(device))
                else:
                    setattr(self, key, value)

        # convert particles to tensor
        self.particles = torch.Tensor(self.particles).to(device)

        """ random gp model"""
        self.hyper_post = HyperPosterior(
            input_dim=self.input_dim,
            feature_dim=self.feature_dim,
            prior_factor=None,
            hyper_prior_dict={
                'lengthscale_raw_loc': 0,'lengthscale_raw_scale': 1,
                'variance_raw_loc': 0,'variance_raw_scale': 1,
                'outputscale_raw_loc': 0,'outputscale_raw_scale': 1,
                'noise_raw_loc': 0, 'noise_raw_scale': 1,
                'constant_mean_loc':0, 'constant_mean_scale':1e-3}, # a dummy hyper-prior, doesn't matter
            covar_module_str=serialized_mdl['covar_module_str'],
            mean_module_str=serialized_mdl['mean_module_str'],
            mean_nn_layers=serialized_mdl['mean_nn_layers'],
            kernel_nn_layers=serialized_mdl['kernel_nn_layers'],
            nonlinearity_hidden_m=serialized_mdl['nonlinearity_hidden_m'],
            nonlinearity_hidden_k=serialized_mdl['nonlinearity_hidden_k'],
            nonlinearity_output_m=serialized_mdl['nonlinearity_output_m'],
            nonlinearity_output_k=serialized_mdl['nonlinearity_output_k'],
            likelihood_str=serialized_mdl['likelihood_str'],
            optimize_noise=self.optimize_noise, noise_std=self.noise_std,
            optimize_lengthscale=serialized_mdl['optimize_lengthscale'],
            lengthscale_fix=serialized_mdl['lengthscale_fix']
        )



    def meta_fit():
        raise NotImplementedError




    def predict_only_mean(self, context_x, context_y, test_x):
        """
        predictions made only using the GP mean
        """
        context_x, context_y = _handle_input_dimensionality(context_x, context_y)
        test_x = _handle_input_dimensionality(test_x)
        assert test_x.shape[1] == context_x.shape[1]
        # normalize data and convert to tensor
        context_x, context_y = self._prepare_data_per_task(context_x, context_y)
        test_x = self._normalize_data(X=test_x, Y=None)
        test_x = torch.from_numpy(test_x).float().to(device)
        pred = []
        with torch.no_grad():
            for particle_num in np.arange(self.num_particles):
                gp_fn = self.hyper_post.get_forward_fn(self.particles[particle_num, :])
                mean_fn = gp_fn.mean_nn
                pred.append(mean_fn.forward(test_x).cpu().detach().numpy())
            pred = sum(pred)/self.num_particles*self.y_std+self.y_mean
        return pred, 0 # TODO: compute var


    """ define predictive dist """

    def get_pred_dist(self, x_context, y_context, x_valid):
        with torch.no_grad():
            x_context = x_context.view(torch.Size((1,)) + x_context.shape).repeat(self.num_particles, 1, 1)
            y_context = y_context.view(torch.Size((1,)) + y_context.shape).repeat(self.num_particles, 1)
            x_valid = x_valid.view(torch.Size((1,)) + x_valid.shape).repeat(self.num_particles, 1, 1)

            gp_fn = self.hyper_post.get_forward_fn(self.particles)
            gp, likelihood = gp_fn(x_context, y_context, train=False)
            pred_dist = likelihood(gp(x_valid))
        return pred_dist



# ----- SERIALIZE MODEL FUNCTION -----
def serialize_model(gp_model, normalize_data, random_seed):
    serialized_mdl={'normalize_data': normalize_data,
                    'random_seed': random_seed}
    # info from GPR_meta_SVGD
    serialized_mdl['particles'] = gp_model.particles.cpu().detach().numpy() # num_rows = num_particles, columns are prior params
    serialized_mdl['num_particles'] = serialized_mdl['particles'].shape[0]
    serialized_mdl['y_mean'], serialized_mdl['y_std'] = gp_model.y_mean, gp_model.y_std
    serialized_mdl['x_mean'], serialized_mdl['x_std'] = gp_model.x_mean, gp_model.x_std
    serialized_mdl['input_dim'], serialized_mdl['feature_dim'] = gp_model.input_dim, gp_model.feature_dim
    serialized_mdl['noise_std'] = gp_model.noise_std
    serialized_mdl['optimize_noise'] = gp_model.optimize_noise
    serialized_mdl['optimize_lengthscale'] = gp_model.optimize_lengthscale
    serialized_mdl['lengthscale_fix'] = gp_model.lengthscale_fix

    # info from random gp.gp
    serialized_mdl['nonlinearity_hidden_m'] = gp_model.hyper_post.gp.nonlinearity_hidden_m
    serialized_mdl['nonlinearity_output_m'] = gp_model.hyper_post.gp.nonlinearity_output_m
    serialized_mdl['nonlinearity_hidden_k'] = gp_model.hyper_post.gp.nonlinearity_hidden_k
    serialized_mdl['nonlinearity_output_k'] = gp_model.hyper_post.gp.nonlinearity_output_k
    serialized_mdl['mean_module_str']  = gp_model.hyper_post.gp.mean_module_str
    serialized_mdl['covar_module_str'] = gp_model.hyper_post.gp.covar_module_str
    serialized_mdl['mean_nn_layers']  = gp_model.hyper_post.gp.mean_nn_layers
    serialized_mdl['kernel_nn_layers'] = gp_model.hyper_post.gp.kernel_nn_layers
    serialized_mdl['likelihood_str'] = gp_model.hyper_post.gp.likelihood_str
    # fit info
    serialized_mdl['fitted'], serialized_mdl['over_fitted'], serialized_mdl['non_psd_cov'] = gp_model.fitted, gp_model.over_fitted, gp_model.non_psd_cov
    # history
    serialized_mdl['history'] = gp_model.history
    # return dict
    return serialized_mdl






if __name__ == "__main__":
    # ----- Import packages -----
    import random, pickle, os, sys, torch
    import numpy as np
    from numpy import random

    from experiments.data_sim import VehicleDataset
    from server.GPR_meta_svgd import GPRegressionMetaLearnedSVGD

    random_seed = 3
    random.seed(random_seed)
    np.random.seed(random_seed)
    random_state = np.random.RandomState(random_seed)

    exp_name='vehicle_linear'
    # ----- LOAD DATA -----
    env = VehicleDataset(random_state=random_state)
    clients_data = env.generate_clients_data(min_n_obs = 200, max_n_obs=200, min_n_test=200, max_n_test=200)
    num_clients = env.num_clients
    meta_train_data = []
    meta_test_data = []
    for n in np.arange(num_clients):
        x_obs, y_obs, x_tru, y_tru = clients_data[n]
        meta_train_data.append((x_obs, y_obs))
        meta_test_data.append((x_tru, y_tru))

    # ----- TRAIN MODELS -----
    torch.set_num_threads(8)
    beta=clients_data[0][0].shape[0]
    # hyper- prior # TODO: refine
    hyper_prior_dict = {'lengthscale_raw_loc': 1,'lengthscale_raw_scale': 5}
    normalize_data=False
    gp_model_svgd = GPRegressionMetaLearnedSVGD(meta_train_data,
                                                num_iter_fit=1500, feature_dim=1,
                                                prior_factor=0.6,
                                                hyper_prior_dict=hyper_prior_dict,
                                                covar_module_str='linear',
                                                mean_module_str='constant',
                                                kernel_nn_layers=None,
                                                mean_nn_layers=None,
                                                optimizer='Adam', lr=1e-3, lr_decay=0.95,
                                                task_batch_size=5,
                                                normalize_data=normalize_data,
                                                optimize_noise=False, noise_std=float(0.6),
                                                bandwidth=10, kernel='RBF',
                                                random_seed=random_seed, num_particles=1,
                                                logger=None)

    gp_model_svgd.meta_fit(valid_tuples=clients_data, over_fit_margin=1e-3, verbose=True,
                            cont_fit_margin =1e-3, max_iter_fit=3000, log_period=500)
    # evaluate trained model
    valid_ll, valid_rmse, valid_calibr = gp_model_svgd.eval_datasets(clients_data)

    # ----- SAVE MODEL -----
    serialized_mdl = serialize_model(gp_model_svgd, normalize_data=normalize_data, random_seed=random_seed)
    # save
    filename_save = "test_model"
    file = open(filename_save, 'wb')
    pickle.dump(serialized_mdl, file)
    file.close()


    # ----- SAVE DATA -----
    filename_data = "test_data"
    file = open(filename_data, 'wb')
    pickle.dump(clients_data, file)
    file.close()
    # save as .mat
    # scipy.io.savemat(filename_data+'.mat', {'clients_data': clients_data})
    print('[INFO] saved data for {:2.0f} clients'.format(num_clients))


    # delete everything!
    del gp_model_svgd, clients_data

    # load data
    file = open(filename_data, 'rb')
    clients_data = pickle.load(file)
    print('[INFO] environment loaded')

    # load models
    file = open(filename_save, 'rb')
    serialized_mdl = pickle.load(file)
    file.close()

    # reconstruct
    gp_model_svgd = TrainedSVGD(serialized_mdl)

    # evaluate
    valid_ll2, valid_rmse2, valid_calibr2 = gp_model_svgd.eval_datasets(clients_data)

    # compare
    if (abs(valid_ll2-valid_ll2)<1e-6) and (abs(valid_rmse2-valid_rmse)<1e-6) and (abs(valid_calibr2-valid_calibr)<1e-6):
        print('Results match!')
