import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


class RidgeWrapper(Ridge):

    def __init__(self, alphas=None, flatten_y = False, normalize_data=True):
        super().__init__()
        self.flatten_y = flatten_y
        self.normalize_data = normalize_data
        self.alphas = np.logspace(-6,2,80) if alphas is None else alphas

    def fit(self, x_train, y_train):
        # make sure y is 2D
        y_train = np.reshape(y_train, (-1, 1))

        # normalize data
        if self.normalize_data:
            # statistics on train data
            x_mean, y_mean = np.mean(x_train, axis=0), np.mean(y_train, axis=0)
            x_std, y_std = np.std(x_train, axis=0) + 1e-8, np.std(y_train, axis=0) + 1e-8
            # normalize
            x_train_nrm = (x_train - x_mean[None, :]) / x_std[None, :]
            y_train_nrm = (y_train - y_mean[None, :]) / y_std[None, :]
            # check sizes
            assert x_train_nrm.shape == x_train.shape
            assert y_train_nrm.shape == y_train.shape
            x_train, y_train = x_train_nrm, y_train_nrm
        else:
            x_mean, y_mean = np.zeros(x_train.shape[1]), np.zeros(y_train.shape[1])
            x_std, y_std = np.ones(x_train.shape[1]), np.ones(y_train.shape[1])
        self.x_mean, self.y_mean = x_mean, y_mean
        self.x_std, self.y_std = x_std, y_std

        # flatten y if needed
        if self.flatten_y:
            y_train = y_train.flatten()

        # ------ non-regularized lin reg ------
        # train
        best_model = LinearRegression().fit(x_train, y_train)
        # evaluate
        y_train_pred = best_model.predict(x_train).reshape(y_train.shape)
        cur_rmse= ((np.mean((y_train_pred-y_train)**2))**0.5)*y_std[0]
        min_rmse = cur_rmse
        best_alpha = 0


        # ------ regularized lin reg ------
        for alpha in self.alphas:
            # train
            reg = Ridge(alpha=alpha).fit(x_train, y_train)
            # evaluate
            y_train_pred = reg.predict(x_train).reshape(y_train.shape)
            cur_rmse = ((np.mean((y_train_pred-y_train)**2))**0.5)*y_std[0]

            # compare with the best
            if cur_rmse< min_rmse:
                best_model = reg
                min_rmse = cur_rmse
                best_alpha = alpha

        # set attributes
        self.best_model = best_model
        self.coef_ = best_model.coef_
        self.intercept_ = best_model.intercept_
        self.n_features_in_ = best_model.n_features_in_
        if best_alpha > 0:
            self._normalize = best_model._normalize
            self.n_iter_ = best_model.n_iter_
        if best_alpha == 0:
            self.rank_ = best_model.rank_
            self.singular_ = best_model.singular_

        return self

    def predict(self, X):
        # normalize
        X_nrm = (X - self.x_mean[None, :]) / self.x_std[None, :]
        assert X_nrm.shape == X.shape           # check sizes

        # predict
        y_pred = self.best_model.predict(X_nrm)

        # scale back
        if self.flatten_y:
            y_pred_unnorm = y_pred * self.y_std[0] + self.y_mean.flatten()
        else:
            y_pred_unnorm = y_pred * self.y_std[None, :] + self.y_mean[None, :]
        assert y_pred_unnorm.shape == y_pred.shape
        # y_train_nrm = (y_train - y_mean[None, :]) / y_std[None, :]

        return y_pred_unnorm
