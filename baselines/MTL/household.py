import sys, os, copy, math, torch
import numpy as np
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from SMWrapper import SMWrapper
from server import MTL_server


# class SyNet(sy.Module):
#     def __init__(self, torch_ref, in_dim, out_dim):
#         super(SyNet, self).__init__(torch_ref=torch_ref)
#         self.linear = self.torch_ref.nn.Linear(in_dim, out_dim)

#     def forward(self, x):
#         x = self.linear(x)
#         return x

#     def predict(self, X):
#         return self(torch.FloatTensor(X))

class SyNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, random_seed):
        torch.manual_seed(random_seed)
        super(SyNet, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim, dtype=torch.double)

    def forward(self, x):
        x = self.linear(x)
        return x

    def predict(self, X):
        return self(X)

##################################################################################################################
class Household:
    def __init__(self, client_data, client_ts, random_seed, normalize_data=True):
        self.random_seed = random_seed
        # load consumption data
        self.cons_data = client_ts
        # normalize
        self.normalize_data = normalize_data
        if self.normalize_data:
            # statistics on train data
            self.x_mean = np.mean(client_data[0], axis=0)
            self.y_mean = np.mean(client_data[1], axis=0)
            self.x_std = np.std(client_data[0], axis=0) + 1e-8
            self.y_std = np.std(client_data[1], axis=0) + 1e-8
            # normalize
            data_nrm = ((client_data[0] - self.x_mean[None, :]) / self.x_std[None, :],
                        (client_data[1] - self.y_mean[None, :]) / self.y_std[None, :],
                        (client_data[2] - self.x_mean[None, :]) / self.x_std[None, :],
                        (client_data[3] - self.y_mean[None, :]) / self.y_std[None, :])
            # check sizes
            assert data_nrm[0].shape == client_data[0].shape
            assert data_nrm[1].shape == client_data[1].shape
            assert data_nrm[2].shape == client_data[2].shape
            assert data_nrm[3].shape == client_data[3].shape
            client_data = data_nrm


        # done
        self.X = torch.from_numpy(copy.deepcopy(client_data[0])).double()
        self.y = torch.from_numpy(copy.deepcopy(client_data[1].reshape(-1, 1))).double()
        self.X_test = torch.from_numpy(copy.deepcopy(client_data[2])).double()
        self.y_test = torch.from_numpy(copy.deepcopy(client_data[3].reshape(-1, 1))).double()
        # info
        self.info = {'total_samples':self.X.shape[0], 'num_features':self.X.shape[1]}



    ##############################################################################################################
    def train_valid_split(self, **kwargs):
        # total data to use
        N = kwargs.get('N', self.X.shape[0])
        # valid_samples
        if 'valid_samples' in kwargs:
            valid_samples = kwargs.get('valid_samples')
            train_inds = np.arange(N-valid_samples)
            valid_inds  = np.arange(N-valid_samples, N)
        # valid_frac
        if 'valid_frac' in kwargs:
            valid_frac = kwargs.get('valid_frac')
            train_inds = np.arange(round(N*(1-valid_frac)))
            valid_inds  = np.arange(round(N*(1-valid_frac)), N)
        if (not 'valid_frac' in kwargs) and (not 'valid_samples' in kwargs):
            print('[ERROR] please provide a split method')
        if ('valid_frac' in kwargs) and ('valid_samples' in kwargs):
            print('[ERROR] two split methods were provided')
        # Make dataset smaller
        self.X_small = self.X[0:N, :]
        self.y_small = self.y[0:N, :]
        # train and valid sets
        self.X_train = self.X[train_inds, :]
        self.y_train = self.y[train_inds, :]
        self.X_valid  = self.X[valid_inds, :]
        self.y_valid  = self.y[valid_inds, :]
        # has data?
        self.has_train_data = len(self.y_train)>0
        self.has_valid_data  = len(self.y_valid)>0
        # info
        self.info = {'total_samples':N, 'train_samples':len(self.y_train),
        'valid_samples':len(self.y_valid), 'num_features':self.X.shape[1]}


    ##############################################################################################################
    def fit_personal_model(self, method, **kwargs):
        '''
        kwargs: iterations and lr for Adam, verbose for printing the results, noise_var for GP,
        init_params: initial parameters for Adam
        methods: OLS or Adam for lr, gp
        OLS  -> use the Moore-Penrose pseudoinverse to solve the ls problem
        Adam -> train 1-layer NN with Adam optimizer, starting from random weights,
                number of iterations given by 'iterations', and learning rate 'lr'
        '''
        # unpack kwargs
        verbose=kwargs.get('verbose', False)
        iterations=kwargs.get('iterations')
        lr=kwargs.get('lr')
        if (iterations in kwargs) or (lr in kwargs):
            if (not method=='Adam') and (not method=='AdamGP'):
                print('[WARNING] number of iterations or learning rate will not be used.')
        noise_var=kwargs.get('noise_var')
        if 'noise_var' in kwargs:
            tune_noise_var=False
            if (not method=='GP') and (not method=='AdamGP'):
                print('[WARNING] noise variance will not be used.')
        else:
            tune_noise_var=True
        # initial parameters for Adam
        if 'init_state_dict' in kwargs:
            set_init = True
            init_state_dict = kwargs.get('init_state_dict')
        else:
            set_init = False
        # check if training data is available
        if not self.has_train_data:
            print('no train data')
            params = {"linear.weight": torch.tensor([[0]*self.info["num_features"]]).double(),
                      "linear.bias": torch.tensor([0]).double()}
            self.params = params
            return

        # fit
        if method=='OLS':
            self.personal_lr = SMWrapper(sm.OLS).fit(self.X_train, self.y_train)
            self.params = self.personal_lr.fitted_model_.params # params[0] is the intercept

        if method=='Adam':
            model = SyNet(#torch,
                in_dim=self.info['num_features'] , out_dim=1, random_seed=self.random_seed)
            # initial parameters
            if set_init:
                for key, value in init_state_dict.items():
                    model.state_dict()[key].copy_(value.double())
            optim = torch.optim.Adam(params=model.parameters(),lr=lr)
            # iterate
            if verbose:
                print('[INFO] losses are printed for evaluation but are not used by the operator')
            for i in range(iterations):
                optim.zero_grad()
                # predict
                output = model(torch.FloatTensor(self.X_train))
                # calculate loss
                loss = torch.nn.functional.mse_loss(output, self.y_train.reshape(-1, 1))
                loss.backward()
                optim.step()
                if verbose:
                    print("Epoch ", i, " train loss", loss.item())
                    print(model.state_dict())
            self.personal_lr = model
            self.params = self.personal_lr.state_dict()

        if method=='GP':
            raise NotImplementedError
            # # Dataset needs to be converted to tensor for GPflow to handle it
            # data  = (torch.from_numpy(self.X_train, dtype=torch.float64),
            #          torch.from_numpy(self.y_train, dtype=torch.float64))

            # # Defining the GP
            # kernel = gpflow.kernels.SquaredExponential()
            # my_gp  = gpflow.models.GPR(data, kernel=kernel)
            # if not tune_noise_var:
            #     set_trainable(my_gp.likelihood.variance, False)
            #     my_gp.likelihood.variance.assign(noise_var)

            # # Picking an optimizer and training the GP through MLE
            # opt = gpflow.optimizers.Scipy()
            # opt.minimize(my_gp.training_loss, my_gp.trainable_variables,
            #              tol=1e-11, options=dict(maxiter=1000), method='l-bfgs-b')

            # # Let's take a look at its hyperparameters (after training)
            # #print_summary(my_gp)
            # self.personal_gp = my_gp

        # LR model + GP for residuals
        if method=='AdamGP':
            raise NotImplementedError
            # # fit LR
            # self.fit_personal_model(method='Adam', iterations=iterations, lr=lr, verbose=verbose)

            # # calculate residuals
            # self.y_pred_lr_train = self.predict(data=self.X_train, model=self.personal_lr, method='Adam')
            # res = self.y_train - self.y_pred_lr_train.reshape(-1, 1)
            # # fit GP
            # data  = (torch.from_numpy(self.X_train, dtypr=torch.float64),
            #         torch.from_numpy(res, dtype=torch.float64))
            # kernel = gpflow.kernels.SquaredExponential()
            # my_gp  = gpflow.models.GPR(data, kernel=kernel)
            # if not tune_noise_var:
            #     set_trainable(my_gp.likelihood.variance, False)
            #     my_gp.likelihood.variance.assign(noise_var)
            # opt = gpflow.optimizers.Scipy()
            # opt.minimize(my_gp.training_loss, my_gp.trainable_variables,
            #              tol=1e-11, options=dict(maxiter=1000), method='l-bfgs-b')
            # self.residual_gp = my_gp


    ##############################################################################################################
    def minibatch_SGD(self, model, optim, **kwargs):
        '''
        kwargs: mini batch size (mbsize), verbose for printing the results
        '''
        # unpack kwargs
        verbose=kwargs.get('verbose', False)
        mbsize=kwargs.get('mbsize')
        # check if training data is available
        if not self.has_train_data:
            print('no train data')
            return

        # initialize param update
        cur_state_dict=copy.deepcopy(model.state_dict())

        # iterate
        for i in np.arange(mbsize):
            optim.zero_grad()
            # predict
            output = model(self.X_train)
            # calculate loss
            loss = torch.nn.functional.mse_loss(output, self.y_train.reshape(-1, 1))
            loss.backward()
            optim.step()
            if verbose and i%10==0:
                print("Epoch ", i, " train loss", loss.item())
        # calculate change
        delta_weight = model.state_dict()['linear.weight'].numpy().flatten()-cur_state_dict['linear.weight'].numpy().flatten()
        delta_bias = model.state_dict()['linear.bias'].numpy() - cur_state_dict['linear.bias'].numpy()
        return delta_bias, delta_weight


    ##############################################################################################################
    def predict(self, data, method, **kwargs):
        '''
        kwargs: model (use personal model if not provided)
        '''
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).double()
        if not data.shape[1] == self.info['num_features']:
            print('[ERROR] number of features doea not match the training data')
            return
        if data.shape[0]==0:
            return
        if method=='OLS':
            model = kwargs.get('model', self.personal_lr)
            return model.predict(data)
        if method=='Adam':
            model = kwargs.get('model', self.personal_lr)
            return model(data).data.numpy().flatten()
        if method=='MTL':
            model = kwargs.get('model', self.model_mtl)
            return model(data).data.numpy().flatten()
        if method=='GP':
            if 'model' in kwargs:
                model = kwargs.get('model')
            else:
                model = self.personal_gp
            mean, var = model.predict_f(data)
            return mean[:, 0].numpy(), var[:, 0].numpy()
        if method=='AdamGP':
            model_lr = kwargs.get('model_lr', self.personal_lr)
            model_gp = kwargs.get('model_gp', self.residual_gp)
            pred_lr      = self.predict(data=data, method='Adam', model=model_lr)
            pred_gp, var = self.predict(data=data, method='GP',   model=model_gp)
            return pred_lr+pred_gp, var


    ##############################################################################################################
    def evaluate_model(self, method, measures=['MSE'], verbose=False, **kwargs):
    # ALWAYS USE FLATTEN BEFORE CALCULATING ERRORS
        # initialize dict
        res = {'MSE_train':-1, 'MSE_valid':-1, 'MSE_test':-1}

        # errors
        for ttv in ['train', 'test', 'valid']:
            if ttv=='train':
                X_data, y_data = self.X_train, self.y_train
            if ttv=='valid':
                X_data, y_data = self.X_valid, self.y_valid
            if ttv=='test':
                X_data, y_data = self.X_test, self.y_test
            # predict
            if method=='OLS' or method=='Adam':
                y_pred = self.predict(X_data, method=method, **kwargs)
            if method=='GP' or method=='AdamGP':
                y_pred, var_pred = self.predict(data=X_data, method=method, **kwargs)

            # calculate error measures
            y = y_data.flatten()
            for meas in measures:
                if meas=='MSE':
                    res['MSE_'+ttv] = np.mean((y_pred-y)**2)
                if meas=='MAE':
                    res['MAE_'+ttv] = mean_absolute_error(y, y_pred)
                if meas=='R2':
                    res['R2_'+ttv]  = r2_score(y, y_pred)
                if meas=='Adjr2':
                    n=X_data.shape[0]
                    p=self.info['num_features']
                    res['Adjr2_'+ttv] = 1-(1-res['R2_'+ttv])*(n-1)/(n-p-1)
                if meas=='AIC':
                    res['AIC_'+ttv] = -2*math.log(len(y)*res['MSE_'+ttv])+2*p

        # print
        if verbose:
            for meas in measures:
                if meas=='MAE':
                    print('Mean absolute error: train %.2f, valid %.2f, test %.2f' % (res['MAE_train'], res['MAE_valid'], res['MAE_test']))
                if meas=='MSE':
                    print('Mean squared error:  train %.2f, valid %.2f, test %.2f' % (res['MSE_train'], res['MSE_valid'], res['MSE_test']))
                if meas=='R2':
                    print('Coefficient of determination (R2): train %.2f, valid %.2f, test %.2f' %(res['R2_train'], res['R2_valid'], res['R2_test']))
                if meas=='Adjr2':
                    print('Adjusted coeff. of determination:  train %.2f, valid %.2f, test %.2f' %(res['Adjr2_train'], res['Adjr2_valid'], res['Adjr2_test']))
                if meas=='AIC':
                    print('AIC: train %.2f, valid %.2f, test %.2f' %(res['AIC_train'], res['AIC_valid'], res['AIC_test']))
        return res


    def mtl_init(self, lr):
        # check if training data is available
        if not self.has_train_data:
            print('no train data')
            return
        # initial model
        self.model_mtl = SyNet(#torch,
            in_dim=self.info['num_features'], out_dim=1, random_seed=self.random_seed)
        # initialize optimizer
        self.optim_mtl = torch.optim.Adam(params=self.model_mtl.parameters(),lr=lr)
        return


    def mtl_iterate(self, w_0_wght, w_0_bias, inner_iters, lambda_, verbose):
        # get current parameters
        cur_state_dict=copy.deepcopy(self.model_mtl.state_dict())

        # iterate
        for i in np.arange(inner_iters):
            self.optim_mtl.zero_grad()
            # prediction loss
            output = self.model_mtl(self.X_train)
            loss = torch.nn.MSELoss()(output, self.y_train.reshape(-1, 1))

            # penalty
            l2_reg = torch.tensor(0.).double()
            l2_reg += torch.square(self.model_mtl.linear.weight.flatten()-w_0_wght).sum().sum()
            l2_reg += torch.square(self.model_mtl.linear.bias.flatten()-w_0_bias).sum().sum()
            #l2_reg += torch.norm(self.model_mtl.parameters()[0]-w_0_wght)
            #l2_reg += torch.norm(self.model_mtl.parameters()[1]-w_0_bias)
            loss += lambda_ * l2_reg

            loss.backward()
            self.optim_mtl.step()
            if verbose and i%10==0:
                print("Epoch ", i, " train loss", loss.item())
        # calculate change
        delta_weight = self.model_mtl.state_dict()['linear.weight'].numpy().flatten()-cur_state_dict['linear.weight'].numpy().flatten()
        delta_bias = self.model_mtl.state_dict()['linear.bias'].numpy() - cur_state_dict['linear.bias'].numpy()
        return delta_bias, delta_weight