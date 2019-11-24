#!/usr/bin/python3
"""
Tests the our hyperparameter optimization

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

import unittest
import random
import numpy as np
import rmm.hyperopt as hyperopt

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'


# Linear model class for the unit test below
class LinearModel:

    def __init__(self, slope):
        self.slope = slope

    def predict(self, X):
        return self.slope * X

# training function for the linear model class
def train_linear_model(params, Xs, Ys):
    # ignore the data and just set the slope
    return LinearModel(**params)

class TestHyperopt(unittest.TestCase):

    def test_hyperopt(self):
        # test whether our generic hyperparameter optimization technique
        # is able to find the best hyperparameter setting in a simple case

        # in particular, our model is a linear model where we treat the
        # slope as hyperparameter

        # generate training data with a ground truth slope
        num_folds = 3
        num_seqs_per_fold = 5
        seq_len = 10

        ground_truth_slope = 2.

        data = []

        for f in range(num_folds):
            Xs = []
            Ys = []
            for j in range(num_seqs_per_fold):
                # generate the input values at random
                X = np.random.randn(seq_len, 1)
                Y = ground_truth_slope * X
                Xs.append(X)
                Ys.append(Y)
            data.append((Xs, Ys))

        param_ranges = {'slope' : [0., 1., 2., 3., 4.]}
        R = 30

        model, param_opt, params, errors = hyperopt._hyperopt(train_linear_model, param_ranges, data, R)

        # ensure that the best model has the right slope
        self.assertEqual(ground_truth_slope, model.slope)

        expected_params = {'slope' : ground_truth_slope}
        self.assertEqual(expected_params, param_opt)

        # ensure that all parameter settings with the best slope had the lowest
        # error
        mean_errs = np.mean(errors, axis=1)
        for rmin in np.where(mean_errs < np.min(mean_errs) + 1E-3)[0]:
            self.assertEqual(expected_params, params[rmin])

    def test_hyperopt_esn(self):
        # test hyperparameter optimization for ESNs on the delayed impulse
        # response task

        # generate input data
        T = 300
        delay = 100
        num_folds = 3
        num_seqs_per_fold = 3

        data = []

        for f in range(num_folds):
            Xs = []
            Ys = []
            for j in range(num_seqs_per_fold):
                # sample the input pulse time
                t_offset = random.randrange(20, delay)
                # generate input sequence
                X = np.zeros((T, 1))
                X[t_offset, :] = 1.
                # generate output sequence
                Y = np.zeros((T, 1))
                Y[t_offset + delay, :] = 1.
                Xs.append(X)
                Ys.append(Y)
            data.append((Xs, Ys))

        # set possible hyperparameters for the esn
        ms   = [16, 256]
        w_cs = [0.1, 0.99]
        w_js = [0.]
        input_normalizations = [False]

        R = 20

        # perform hyperparameter optimization
        model, param_opt, params, errors = hyperopt.hyperopt_esn(data, R, ms = ms, w_cs = w_cs, w_js = w_js, input_normalizations = input_normalizations)

        # ensure that we achieved a low error
        mean_errs = np.mean(errors, axis=1)
        self.assertTrue(np.min(mean_errs) < 1E-3, 'expected good task results but got error of %g' % np.min(mean_errs))

        # ensure that the best model has the right parameters
        self.assertEqual(256, model.m)
        self.assertEqual(0.99, model.w_c)
        self.assertEqual(256, param_opt['m'])
        self.assertEqual(0.99, param_opt['w_c'])


    def test_hyperopt_rmm(self):
        # test hyperparameter optimization for AlignRMMs.


        # generate input data
        num_folds = 3
        num_seqs_per_fold = 10
        max_len = 10
        n = 8

        data = []

        for f in range(num_folds):
            Xs = []
            Ys = []
            for j in range(num_seqs_per_fold):
                # start by sampling the sequence length
                T = random.randrange(1, max_len+1)
                # initialize the input and output sequence
                X = np.zeros((2*T+2, n+1))
                Y = np.zeros((2*T+2, n))
                # fill the input sequence with random bits
                X[1:T+1, :n] = np.round(np.random.rand(T, n))
                # add the end of sequence token
                X[T+1, n]  = 1.
                # and copy the sequence to the output
                Y[T+2:, :] = X[1:T+1, :n]
                # append to dataset
                Xs.append(X)
                Ys.append(Y)
            data.append((Xs, Ys))

        # set possible hyperparameters for the AlignRMM
        ms = [2, 128]
        Ks = [16]
        vs = [0.1]
        w_cs = [0.8]
        w_js = [0.3]
        controller_initializations = ['identity']
        # if we do not permit duplicates, this task should not work
        permit_duplicatess = [True]
        input_normalizations = [False]

        R = 10

        # perform hyperparameter optimization
        model, param_opt, params, errors = hyperopt.hyperopt_rmm(data, R, ms = ms, vs=vs, w_cs = w_cs, w_js = w_js, Ks = Ks, controller_initializations = controller_initializations, permit_duplicatess = permit_duplicatess, input_normalizations = input_normalizations)

        # ensure that we achieved a low error
        mean_errs = np.mean(errors, axis=1)
        self.assertTrue(np.min(mean_errs) < 3., 'expected good task results but got error of %g' % np.min(mean_errs))

        # ensure that the best model has the right parameters
        self.assertEqual(128, model.m)
        self.assertEqual(128, param_opt['m'])


if __name__ == '__main__':
    unittest.main()
