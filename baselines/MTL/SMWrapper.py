import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.metrics import r2_score
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import KFold



class SMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept


    def fit(self, X, y,
            feat_names=[]):
        # initialize
        if self.fit_intercept:
            X = sm.add_constant(X)
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.model_ = self.model_class(y, X)
        # fit
        self.fitted_model_ = self.model_.fit()
        self.feature_importances_ = self.fitted_model_.tvalues
        # optional
        if not feat_names==[]:
            self.feat_names_=feat_names
        return self


    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, 'model_')
        # Input validation
        X = check_array(X)
        # predict
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.fitted_model_.predict(X)


    def get_params(self, deep = False):
        return {'fit_intercept':self.fit_intercept,
               'model_class':self.model_class}


    def summary(self, xname=[]):
        return self.fitted_model_.summary(xname=xname)


    def cv_fit(self, X, y, n_splits=5,
               scoring='r2', n_jobs=-1,
               return_train_score=False,
               return_estimator=True,
               verbose=True):
        r2_train = [0]*n_splits
        r2_test = [0]*n_splits
        models = []
        test_inds = []
        # folds
        kf = KFold(n_splits=n_splits, shuffle=True)
        fold_num=0
        for train_index, test_index in kf.split(X):
            test_inds.append(test_index)
            X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
            y_train, y_test = y[train_index], y[test_index]
            # fit to this fold
            self.fit(X_train, y_train)
            models.append(self.fitted_model_)
            r2_train[fold_num] = self.fitted_model_.rsquared
            # test scores
            y_test_pred = self.predict(X_test)
            r2_test[fold_num] = r2_score(y_test, y_test_pred)
            fold_num = fold_num+1
        # find best
        if verbose:
            print('Train R2 scores: ' + str(r2_train)[1:-1])
            print('Test  R2 scores: ' + str(r2_test)[1:-1])
        maxr2, ind = max((val, idx) for (idx, val) in enumerate(r2_test))
        self.fitted_model_ = models[ind]
        self.test_inds_ = test_inds[ind]
        self.train_inds_ = np.setdiff1d(np.arange(0, len(X.iloc[:,0])),self.test_inds_)
        return self
