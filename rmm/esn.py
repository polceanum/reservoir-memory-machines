"""
Provides echo state network classes according to the scikit-learn
interface.

Copyright (C) 2019
Benjamin Paaßen, Alexander Schulz
AG Machine Learning
Bielefeld University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import rmm.crj as crj

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen, Alexander Schulz'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'


def linreg(H, Y, regul):
    """ Applies linear regression to map the representation time series
    Phi to the output time series Y.

    Args:
    H: A T x m representation time series.
    Y: A T x K output time series.

    Return:
    V:   A K x m matrix mapping representation to output dimensions,
         such that np.dot(H, V.T) is as similar as possible to Y.
    """
    C    = np.dot(H.T, H)
    Cinv = np.linalg.inv(C + regul * np.eye(H.shape[1]))
    Cout = np.dot(Y.T, H)
    return np.dot(Cout, Cinv)

class ESN(BaseEstimator, RegressorMixin):
    """ Implements an echo state network (Jaeger and Haas, 2004) with cycle
    reservoir with jumps (Rodan and Tino, 2012) as basis.

    Attributes:
    m:   The number of neurons in the network. 128 per default.
    v:   The input-to-reservoir connection strength (should be a float larger
         than 0). 0.1 per default.
    w_c: The connection strength along the cycle. Needs to be in the range
         (0, 1). Smaller numbers mean less memory and faster forgetting.
         0.9 per default.
    w_j: The connection strength along jumps. Needs to be in the range (0, 1).
         Smaller numbers mean less memory and faster forgetting.
         0.3 per default.
    l:   The length of jumps along the cycle. 10% of m per default.
    leak: The 'leak rate' alpha, i.e. what fraction of the current memory
         content is overriden by the new values. This parameter can be used
         to 'smoothen' the reaction of the net, i.e. if the time scale of the
         output curve is much lower than the time scale of the input curve.
    regul: The L2 regularization strength for linear regression. 1E-5 per default.
    input_normalization: Switch to true if the input should be z-normalized
         before feeding it into the network. True per default.
    washout: The number of washout steps before training. 0 per default.
    _U:  The input-to-reservoir matrix. Is set during training.
    _W:  The reservoir connection matrix as a scipy sparse csr matrix.
         Is set during training.
    _V:  The reservoir-to-output matrix. Is set during training.
    _mu: The mean of the training data for each feature. This is used for
         normalization purposes.
    _beta: The precision of the training data for each feature.
         This is used for normalization purposes.
    """
    def __init__(self, m = 128, v = .1, w_c = 0.9, w_j = 0.3, l = None, leak = 1., regul = 1E-5, input_normalization = True, washout = 0 ):
        self.m = m
        self.v = v
        self.w_c = w_c
        self.w_j = w_j
        if l is None:
            if m > 100:
                self.l = int(m / 10)
            elif m > 10:
                self.l = int(m / 5)
            else:
                self.l = int(m / 2)
        else:
            self.l = l
        self.leak = leak
        self.regul = regul
        self.input_normalization = input_normalization
        self.washout = 0

    def _init_reservoir(self, n):
        """ Initializes the reservoir for this ESN """
        self._U = crj.setup_input_weight_matrix(n, self.m, self.v)
        self._W = crj.setup_reservoir_matrix(self.m, self.w_c, self.w_j, self.l)

    def _apply_reservoir(self, X):
        """ Applies this networks reservoir to the given normalized input time series. """
        T = X.shape[0]
        # initialize output matrix
        H = np.zeros((T, self.m))
        # compute retain rate
        retain = 1. - self.leak
        if retain < 1E-3:
            retain = 0.
        # compute first state
        H[0, :] = np.tanh(np.dot(self._U, X[0, :]))
        # compute remaining states
        for t in range(1, T):
            H[t, :] = self.leak * np.tanh(np.dot(self._U, X[t, :]) + self._W * H[t-1, :]) + retain * H[t-1, :]
        return H

    def fit(self, X, Y):
        """ Fits this echo state network to the given data.

        Args:
        X: A T x n input time series matrix OR a list of such matrices.
        Y: A T x K output time series matrix OR a list of such matrices.
        """
        # prepare the input matrix
        if isinstance(X, list):
            Xs = X
        else:
            Xs = [X]
        # prepate input normalization
        n = Xs[0].shape[1]
        self._mu   = np.zeros(n)
        self._beta = np.zeros(n)
        if self.input_normalization:
            # compute the mean first
            T = 0
            for X in Xs:
                T += len(X)
                self._mu += np.sum(X, axis=0)
            self._mu /= T
            self._mu = np.expand_dims(self._mu, 0)
            # then compute precision
            for X in Xs:
                self._beta += np.sum(np.square(X - self._mu), axis=0)
            self._beta[self._beta < 1E-3] = T
            self._beta = np.expand_dims(np.sqrt(T / self._beta), 0)
        else:
            self._mu = 0.
            self._beta = 1.

        # initialize the reservoir
        self._init_reservoir(n)

        # process all input matrices with a cycle reservoir with jumps
        # and concatenate them to one big output
        Hs = []
        for X in Xs:
            H = self._apply_reservoir((X - self._mu) * self._beta)
            # remove washout steps before recording the data
            Hs.append(H[self.washout:, :])
        H = np.concatenate(Hs, axis=0)

        # prepate the output
        if isinstance(Y, list):
            Ys = Y
        else:
            Ys = [Y]
        for j in range(len(Ys)):
            # remove the washout steps from each output matrix
            Ys[j] = Ys[j][self.washout:, :]
        Y = np.concatenate(Ys, axis=0)

        # after this preparation, we can perform linear regression to
        # generate the reservoir-to-output matrix
        self._V = linreg(H, Y, self.regul)

        return self

    def predict(self, X):
        """ Predicts on the given input time series using this ESN.

        Args:
        X: A T x n time series.

        Returns:
        Y: A T x K time series.
        """
        # compute the CRJ representation after input normalization
        H = self._apply_reservoir((X - self._mu) * self._beta)
        # apply the output matrix.
        Y = np.dot(H, self._V.T)
        # return
        return Y

class ESGRU(ESN):
    """ Implements an echo state gated recurrent unit, following the dynamic
    equations of Cho et al. (2014) with cycle reservoir with jumps
    (Rodan and Tino, 2012) as basis.

    Attributes:
    m:   The number of neurons in the network. 128 per default.
    v:   The input-to-reservoir connection strength (should be a float larger
         than 0). 0.1 per default.
    w_c: The connection strength along the cycle. Needs to be in the range
         (0, 1). Smaller numbers mean less memory and faster forgetting.
         0.9 per default.
    w_j: The connection strength along jumps. Needs to be in the range (0, 1).
         Smaller numbers mean less memory and faster forgetting.
         0.3 per default.
    l:   The length of jumps along the cycle. 10% of m per default.
    v_gate: The strength of connections to the gates. 0.9 per default.
    regul: The L2 regularization strength for linear regression. 1E-5 per default.
    input_normalization: Switch to true if the input should be z-normalized
         before feeding it into the network. True per default.
    washout: The number of washout steps before training. 0 per default.
    _U:  The input-to-reservoir matrix. Is set during training.
    _W:  The reservoir connection matrix as a scipy sparse csr matrix.
         Is set during training.
    _V:  The reservoir-to-output matrix. Is set during training.
    _mu: The mean of the training data for each feature. This is used for
         normalization purposes.
    _beta: The precision of the training data for each feature.
         This is used for normalization purposes.
    """
    def __init__(self, m = 128, v = .1, w_c = 0.9, w_j = 0.3, l = None, v_gate = 0.9, regul = 1E-5, input_normalization = True, washout = 0 ):
        super(ESGRU, self).__init__(m, v, w_c, w_j, l, 1., regul, input_normalization, washout)
        self.v_gate = v_gate

    def _init_reservoir(self, n):
        """ Initializes the reservoir for this ES-GRU """
        self._U   = crj.setup_input_weight_matrix(n, self.m, self.v)
        self._W   = crj.setup_reservoir_matrix(self.m, self.w_c, self.w_j, self.l)
        self._U_r = crj.setup_input_weight_matrix(n, self.m, self.v_gate, self.m * n)
        self._W_r = crj.setup_input_weight_matrix(self.m, self.m, self.v_gate, 2 * self.m * n)
        self._b_r = crj.setup_input_weight_matrix(1, self.m, self.v_gate, self.m * (2 * n + self.m))[0, :]
        self._U_z = crj.setup_input_weight_matrix(n, self.m, self.v_gate, self.m * (2 * n + self.m + 1))
        self._W_z = crj.setup_input_weight_matrix(self.m, self.m, self.v_gate, self.m * (3 * n + self.m + 1))
        self._b_z = crj.setup_input_weight_matrix(1, self.m, self.v_gate, self.m * (2 * n + 2 * self.m + 1))[0, :]

    def _apply_reservoir(self, X):
        """ Applies this networks reservoir to the given normalized input time series. """
        T = X.shape[0]
        # initialize output matrix
        H = np.zeros((T, self.m))
        # compute first state
        H[0, :] = 0.5 * np.tanh(np.dot(self._U, X[0, :]))
        # compute remaining states
        for t in range(1, T):
            # compute reset gate
            r = np.dot(self._U_r, X[t, :]) + np.dot(self._W_r, H[t-1, :]) + self._b_r
            r = 1. / (1. + np.exp(-r))
            # compute update vector
            htilde = np.tanh(np.dot(self._U, X[t, :]) + self._W * (r * H[t-1, :]))
            # compute update gate
            z = np.dot(self._U_z, X[t, :]) + np.dot(self._W_z, H[t-1, :]) + self._b_z
            z = 1. / (1. + np.exp(-z))
            # compute next state
            H[t, :] = z * htilde + (1. - z) * H[t-1, :]
        return H
