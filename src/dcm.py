import pandas as pd
import numpy as np

from scipy.optimize import minimize
from scipy.stats import norm

import warnings

#ignore by message
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

class DCM():

    name = 'DCM'

    def __init__(self, init_coef=1, const=True, maxiter=1000, cutoff=0.5,
                 beta_init=None, lambda_=0, normalization=False, method='BFGS',
                 bounds=None):

        self.beta_init = beta_init

        self.init_coef = init_coef
        self.const = const
        self.maxiter = maxiter
        self.cutoff = cutoff
        self.lambda_ = lambda_
        self.normalization = normalization
        self.method = method
        self.bounds = bounds

        return None

    def information_matrix(self, X, beta_hat):

        p = self.get_cdf(X, beta_hat)
        g = self.get_pdf(X, beta_hat)

        inf_mat = np.zeros((self.m, self.m))

        for i in range(len(inf_mat)):
            inf_mat[i, i] = (np.square(g[i]) / (p[i] * (1 - p[i]))) * np.dot(X[i, :], X[i, :].T)

        return inf_mat

    def get_z_values(self):

        z_values = np.zeros((len(self.se_beta), 1))

        for i in range(len(self.beta_hat)):
            z_values[i] = self.beta_hat[i] / self.se_beta[i]

        return z_values.flatten()

    def loglikelihood_function(self, beta, X, y):


        h = self.get_cdf(X, beta)

        if self.normalization:
            normalized_beta = beta * (1 / 1 + np.std(beta))
            regular_term = (self.lambda_ / 2) * np.sum(normalized_beta)
        else:
            regular_term = (self.lambda_ / 2) * np.sum(beta)

        llik = -(1 / self.n) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) + regular_term

        return llik

    def optimizer(self, initial_beta, X, y):

        opt = minimize(fun=self.loglikelihood_function,
                            x0=initial_beta,
                            method=self.method,
                            args=(X, y),
                            bounds=self.bounds,
                            options={'maxiter':self.maxiter}
                            )

        return opt

    def get_McFaddenR2(self, y):

        X_const = np.ones((self.n, 1))
        beta_0 = np.array([1 * self.init_coef])
        opt = self.optimizer(beta_0, X_const, y)
        beta_hat_0 = opt.x

        self.llik_0 = self.loglikelihood_function(beta_hat_0, X_const, y)
        self.McFaddenR2 = 1 - (self.llik / self.llik_0)

        return self.McFaddenR2

    def fit(self, X, y):

        self.y = y
        if self.const == True:
            self.X = np.concatenate((np.ones((len(X), 1)), X), axis=1)
            
        else:
            self.X = X

        self.n, self.m = self.X.shape

        if isinstance(self.beta_init, type(None)):
            self.beta = np.ones((self.m, 1)) * self.init_coef
        else:
            self.beta = self.beta_init

        self.opt = self.optimizer(self.beta, self.X, self.y)
        self.beta_hat = self.opt.x
        self.llik = self.opt.fun
        self.inf_mat = np.linalg.inv(self.information_matrix(self.X, self.beta_hat))
        self.se_beta = np.sqrt(np.diag(self.inf_mat))
        self.z_values = self.get_z_values()
        self.AIC = 2 * self.m - 2 * np.log(self.llik)
        self.BIC = np.log(self.n) * self.m - 2 * np.log(self.llik)
        self.McFaddenR2 = self.get_McFaddenR2(self.y)

        return None

    def predict(self, X):

        if self.const == True:
            X = np.concatenate((np.ones((len(X), 1)), X), axis=1)

        self.probs = self.get_cdf(X, self.beta_hat)
        self.preds = np.zeros((len(X),1))

        for i in range(len(self.probs)):
            if self.probs[i] >= self.cutoff:
                self.preds[i] = 1

        return self.preds

    def predict_proba(self, X):

        if self.const == True:
            X = np.concatenate((np.ones((self.n, 1)), X), axis=1)

        return self.get_cdf(X, self.beta_hat)


class Logit(DCM):

    name = 'Logit'

    def get_pdf(self, X, beta):
        z = np.dot(X, beta)
        self.pdf = np.exp(-z) / np.square(1 + np.exp(-z))
        return self.pdf

    def get_cdf(self, X, beta):
        z = np.dot(X, beta)
        self.cdf = 1 / (1 + np.exp(-z))

        return self.cdf

class Probit(DCM):

    name = 'Probit'

    def get_pdf(self, X, beta):
        z = np.dot(X, beta)
        self.pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-(1 / 2) * np.square(z))
        return self.pdf

    def get_cdf(self, X, beta):
        z = np.dot(X, beta)
        self.cdf = 1 / np.square(1 + np.exp(-z))

        for i in range(len(z)):
            self.cdf[i] = norm.cdf(z[i])

        return self.cdf

class Cauchit(DCM):

    name = 'Cauchit'

    def get_pdf(self, X, beta):
        z = np.dot(X, beta)
        self.pdf = 1 / (np.pi * (1 + np.square(z)))
        return self.pdf

    def get_cdf(self, X, beta):
        z = np.dot(X, beta)
        self.cdf = (1 / np.pi) *  (np.arctan(z) + (np.pi/2))

        return self.cdf

class Gompit(DCM):

    name = 'Gompit'

    def get_pdf(self, X, beta):
        z = np.dot(X, beta)
        self.pdf = np.exp(-z-np.exp(-z))
        return self.pdf

    def get_cdf(self, X, beta):
        z = np.dot(X, beta)
        self.cdf = np.exp(-np.exp(-z))

        return self.cdf

class ReverseGompit(DCM):

    name = 'Reverse Gompit'

    def get_pdf(self, X, beta):
        z = np.dot(X, beta)
        self.pdf = np.exp(-z-np.exp(-z))
        return self.pdf

    def get_cdf(self, X, beta):
        z = np.dot(X, beta)
        self.cdf = np.exp(-np.exp(-z))

        return self.cdf


    def revert(self, array):

        for i in range(len(array)):
            if array[i] == 1:
                array[i] = 0
            else:
                array[i] = 1

        return array

    def fit(self, X, y):

        self.y = self.revert(y)
        if self.const == True:
            self.X = np.concatenate((np.ones((len(X), 1)), X), axis=1)

        else:
            self.X = X

        self.n, self.m = self.X.shape

        if isinstance(self.beta_init, type(None)):
            self.beta = np.ones((self.m, 1)) * self.init_coef
        else:
            self.beta = self.beta_init

        self.opt = self.optimizer(self.beta, self.X, self.y)
        self.beta_hat = self.opt.x
        self.llik = self.opt.fun
        self.inf_mat = np.linalg.inv(self.information_matrix(self.X, self.beta_hat))
        self.se_beta = np.sqrt(np.diag(self.inf_mat))
        self.z_values = self.get_z_values()
        self.AIC = 2 * self.m - 2 * np.log(self.llik)
        self.BIC = np.log(self.n) * self.m - 2 * np.log(self.llik)
        self.McFaddenR2 = self.get_McFaddenR2(self.y)

        return self.beta_hat

    def predict(self, X):

        if self.const == True:
            X = np.concatenate((np.ones((self.n, 1)), X), axis=1)

        self.probs = self.get_cdf(X, self.beta_hat)
        self.preds = np.zeros((len(X),1))

        for i in range(len(self.probs)):
            if self.probs[i] >= self.cutoff:
                self.preds[i] = 1

        return self.revert(self.preds)

    def predict_proba(self, X):

        if self.const == True:
            X = np.concatenate((np.ones((self.n, 1)), X), axis=1)

        return 1 - self.get_cdf(X, self.beta_hat)