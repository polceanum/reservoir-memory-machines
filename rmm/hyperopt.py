"""
Implements hyperparameter optimization for all ESN variations
in this package.

Copyright (C) 2019
Benjamin Paaßen
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

import random
import numpy as np
import rmm.esn as esn
import rmm.rmm as rmm

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen, Alexander Schulz'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

def _hyperopt(train_fun, param_ranges, data, R = 10):
    """ Performs hyper-parameter optimization for the given
    training function, the given parameter ranges, and the given
    list of crossvalidation folds. In particular, we generate
    R parameter settings at random and evaluate them in a
    crossvalidation on data.

    Args:
    train_fun: A function which maps a given list of parameters,
               a list of input sequences, and a list of output
               sequences to a trained model, i.e. train_fun(params, Xs, Ys)
               should be a trained odel.
    param_ranges: A dictionary mapping parameter names to lists of possible
               values for each parameter.
    data:      A list of tuples (Xs, Ys), where Xs is a list if input
               sequences and Ys is a list of output sequences.
               This will be used as basis for a crossvalidation.
    R:         The number of random trials for random search. 10 per
               default.

    Returns:
    model:     The best model found.
    param_setting: The best parameter setting found.
    params:    A list of parameter settings with R elements.
    errors:    A R x len(data) matrix of RMSE errors achieved by each
               model.
    """
    # initialize the outputs
    params = []
    errors = np.zeros((R, len(data)))

    # iterate over all random settings
    for r in range(R):
        # generate a random setting of parameters
        param_setting = {}
        for param_name in param_ranges.keys():
            param_range = param_ranges[param_name]
            param_setting[param_name] = param_range[random.randrange(len(param_range))]
        params.append(param_setting)
        # iterate over all crossvalidation folds
        for f in range(len(data)):
            # concatenate the training data
            Xs_train = []
            Ys_train = []
            for f2 in range(len(data)):
                if f == f2:
                    continue
                Xs_train += data[f2][0]
                Ys_train += data[f2][1]
            # train the model
            model = train_fun(param_setting, Xs_train, Ys_train)
            # evaluate the model
            Xs_test = data[f][0]
            Ys_test = data[f][1]
            for j in range(len(Xs_test)):
                Yhat = model.predict(Xs_test[j])
                errors[r, f] = np.sqrt(np.mean(np.sum(np.square(Ys_test[j] - Yhat), axis=1)))
    # select the best parameter setting
    rmin = np.argmin(np.mean(errors, axis=1))
    # accumulate the entire data
    Xs = []
    Ys = []
    for f in range(len(data)):
        Xs += data[f][0]
        Ys += data[f][1]
    # train the model on the entire data
    model = train_fun(params[rmin], Xs, Ys)
    return model, params[rmin], params, errors

def _train_esn(param_setting, Xs, Ys):
    model = esn.ESN(**param_setting)
    model.fit(Xs, Ys)
    return model

def hyperopt_esn(data, R = 10, ms = [128], vs = [0.1], w_cs = [0.9], w_js = [0.3], ls = [None], leaks = [1.], reguls = [1E-5], input_normalizations = [True], washouts = [0]):
    """ Performs hyperparameter optimization for an ESN model on the given
    crossvalidation folds. From the given parameter lists, we choose uniformly
    at random, generating R candidate settings which are evaluated in
    crossvalidation over the given data.

    data:      A list of tuples (Xs, Ys), where Xs is a list if input
               sequences and Ys is a list of output sequences.
               This will be used as basis for a crossvalidation.
    R:         The number of random trials for random search. 10 per
               default.
    ms:        A list of possible number of neurons for the ESN. [128] per
               default.
    vs:        A list of possible input weights. [0.1] per default.
    w_cs:      A list of possible cycle weights. [0.9] per default.
    w_js:      A list of possible jump weights. [0.3] per default.
    ls:        A list of possible jump lengths. [None] per default.
    leaks:     A list of possible leak rates. [1.] per default.
    reguls:    A list of possible regularization strengths. [1E-5] per default.
    input_normalization: A list of possible input normalization settings.
               [True] per default.
    washout:   A list of possible washout lengths. [0] per default.

    Returns:
    model:     The best model found.
    params:    A list of parameter settings with R elements.
    errors:    A R x len(data) matrix of RMSE errors achieved by each
               model.
    """
    param_ranges = {'m' : ms, 'v' : vs, 'w_c' : w_cs, 'w_j' : w_js, 'l' : ls,
        'leak' : leaks, 'regul' : reguls, 'input_normalization' : input_normalizations,
        'washout' : washouts}
    return _hyperopt(_train_esn, param_ranges, data, R)

def _train_esgru(param_setting, Xs, Ys):
    model = esn.ESGRU(**param_setting)
    model.fit(Xs, Ys)
    return model

def hyperopt_esgru(data, R = 10, ms = [128], vs = [0.1], w_cs = [0.9], w_js = [0.3], ls = [None], v_gates = [0.9], bases = [2], steps_selfs = [1], steps_others = [1], reguls = [1E-5], input_normalizations = [True], washouts = [0]):
    """ Performs hyperparameter optimization for an echo state gated recurrent
    unit model model on the given
    crossvalidation folds. From the given parameter lists, we choose uniformly
    at random, generating R candidate settings which are evaluated in
    crossvalidation over the given data.

    data:      A list of tuples (Xs, Ys), where Xs is a list if input
               sequences and Ys is a list of output sequences.
               This will be used as basis for a crossvalidation.
    R:         The number of random trials for random search. 10 per
               default.
    ms:        A list of possible number of neurons for the ESN. [128] per
               default.
    vs:        A list of possible input weights. [0.1] per default.
    w_cs:      A list of possible cycle weights. [0.9] per default.
    w_js:      A list of possible jump weights. [0.3] per default.
    ls:        A list of possible jump lengths. [None] per default.
    v_gates:   A list of possible gate weights. [0.9] per default.
    reguls:    A list of possible regularization strengths. [1E-5] per default.
    input_normalization: A list of possible input normalization settings.
               [True] per default.
    washout:   A list of possible washout lengths. [0] per default.

    Returns:
    model:     The best model found.
    params:    A list of parameter settings with R elements.
    errors:    A R x len(data) matrix of RMSE errors achieved by each
               model.
    """
    param_ranges = {'m' : ms, 'v' : vs, 'w_c' : w_cs, 'w_j' : w_js, 'l' : ls,
        'v_gate' : v_gates,
        'regul' : reguls, 'input_normalization' : input_normalizations,
        'washout' : washouts}
    return _hyperopt(_train_esgru, param_ranges, data, R)

def _train_rmm(param_setting, Xs, Ys):
    model = rmm.RMM(**param_setting)
    model.fit(Xs, Ys)
    return model

def hyperopt_rmm(data, R = 10, ms = [128], vs = [0.1], w_cs = [0.9], w_js = [0.3], ls = [None], leaks = [1.], Ks = [16], controller_initializations = ['identity'], permit_duplicatess = [False], reguls = [1E-5], input_normalizations = [True], washouts = [0]):
    """ Performs hyperparameter optimization for a reservoir memory machine
    on the given crossvalidation folds. From the given parameter lists,
    we choose uniformly at random, generating R candidate settings which are
    evaluated in crossvalidation over the given data.

    data:      A list of tuples (Xs, Ys), where Xs is a list if input
               sequences and Ys is a list of output sequences.
               This will be used as basis for a crossvalidation.
    R:         The number of random trials for random search. 10 per
               default.
    ms:        A list of possible number of neurons for the ESN. [128] per
               default.
    vs:        A list of possible input weights. [0.1] per default.
    w_cs:      A list of possible cycle weights. [0.9] per default.
    w_js:      A list of possible jump weights. [0.3] per default.
    ls:        A list of possible jump lengths. [None] per default.
    leaks:     A list of possible leak rates. [1.] per default.
    Ks:         The number of memory entries. 16 per default.
    controller_initializations: The mechanism for determining the initial
               controller.
               Can be either 'forward' (default) for a policy that always
               moves one step ahead or 'identity' for starting with an
               identity mapping between input and output and determining the
               policy from that point.
    permit_duplicatess: Wether the write head should permit duplicates in memory
               if it helps to maintain linearity in the memory structure.
               [False] per default.
    reguls:    A list of possible regularization strengths. [1E-5] per default.
    input_normalization: A list of possible input normalization settings.
               [True] per default.
    washout:   A list of possible washout lengths. [0] per default.

    Returns:
    model:     The best model found.
    params:    A list of parameter settings with R elements.
    errors:    A R x len(data) matrix of RMSE errors achieved by each
               model.
    """
    param_ranges = {'m' : ms, 'v' : vs, 'w_c' : w_cs, 'w_j' : w_js, 'l' : ls,
        'leak' : leaks,
        'K' : Ks, 'controller_initialization' : controller_initializations,
        'permit_duplicates' : permit_duplicatess,
        'regul' : reguls, 'input_normalization' : input_normalizations,
        'washout' : washouts}
    return _hyperopt(_train_rmm, param_ranges, data, R)
