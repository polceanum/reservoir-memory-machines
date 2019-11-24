#!/usr/bin/python3
"""
Tests the our python implementation of reservoir memory machines.

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
import unittest
import time
import numpy as np
import rmm.esn as esn
import rmm.rmm as rmm

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class TestRMM(unittest.TestCase):

    def test_apply_read(self):
        K = 2
        m = 4
        model = rmm.RMM(m = m, K = K)
        # manually set the controller matrices to read the
        # actions directly from the input stream.
        # The first input dimension determines writes,
        # the remaining dimensions determine reads
        model._V_w = np.zeros((1, 4 + m + 1))
        model._V_w[0, 0] = +1.
        model._V_r = np.zeros((3, 4 + m + 1))
        model._V_r[0, 1] = 1.
        model._V_r[1, 2] = 1.
        model._V_r[2, 3] = 1.
        # set up input and memory trace
        X = np.array([
            [+1., 1., 0., 0.], # write to memory and pause
            [-1., 1., 0., 0.], # don't write to memory and pause
            [-1., 0., 1., 0.], # don't write to memory and step
            [+1., 0., 0., 1.], # write to memory and reset
            [-1., 0., 1., 0.], # don't write to memory and step
            [-1., 0., 1., 0.]  # don't write to memory and step
        ])
        # set up a placeholder reservoir matrix
        H = np.zeros((len(X), model.m))
        # perform reads
        actual_reads = model._apply_read(X, H)
        # compare against expected reads
        expected_reads = np.stack([X[0, :], X[0, :], np.zeros(4), X[0, :], X[3, :], X[0, :]], axis=0)

        np.testing.assert_array_almost_equal(expected_reads, actual_reads)

    def test_align_write_read(self):
        model = rmm.RMM(m = 4, K = 4)
        # manually set the mapping to sum up both input values
        model._V = np.array([[1., 1.]])
        # set up input and output
        X  = np.array([[0., 1.], [0., 0.], [1., 1.], [2., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [1., 2.], [2., 2.]])
        Y  = np.array([[1., 1., 2.2, 2., 1., 1., 2., 2., 2.8, 4.]]).T
        # perform alignment
        writes = model._align_write(X, Y)
        # check write actions against expected actions
        expected_writes = np.array([1., -1., 1., -1., -1., -1., -1., -1., 1., 1.])
        np.testing.assert_array_almost_equal(expected_writes, writes)

        # perform the writes on the input time series and generate
        # an according memory trace
        Ms = np.zeros((len(X), model.K, X.shape[1]))
        write_loc = 0
        for t in range(len(X)):
            if t > 0:
                Ms[t, :, :] = Ms[t-1, :, :]
            if expected_writes[t] > 0.:
                Ms[t, write_loc, :] = X[t, :]
                write_loc += 1
                if write_loc >= model.K:
                    write_loc = 0

        # perform read alignment
        _, reads = model._align_read(Ms, Y)

        # check read actions against expected actions
        expected_reads = np.array([
            [1., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [1., 0., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 1., 0.]])
        np.testing.assert_array_almost_equal(expected_reads, reads)

        # check an example for the copy task with duplicates
        model.permit_duplicates = True
        model._V = np.eye(2, 3)
        model.K = 5
        X = np.array([
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.], 
            [0., 0., 0.],
            [1., 1., 0.],
            [0., 0., 1.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
        ])
        Y = np.array([
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 1.],
            [0., 0.], 
            [0., 0.],
            [1., 1.]
        ])
        # perform alignment
        writes = model._align_write(X, Y)
        # check write actions against expected actions
        expected_writes = np.array([1., 1., 1., 1., 1., -1., -1., -1., -1., -1.])
        np.testing.assert_array_almost_equal(expected_writes, writes)

        # perform the writes on the input time series and generate
        # an according memory trace
        Ms = np.zeros((len(X), model.K, X.shape[1]))
        write_loc = 0
        for t in range(len(X)):
            if t > 0:
                Ms[t, :, :] = Ms[t-1, :, :]
            if expected_writes[t] > 0.:
                Ms[t, write_loc, :] = X[t, :]
                write_loc += 1
                if write_loc >= model.K:
                    write_loc = 0

        # perform read alignment
        _, reads = model._align_read(Ms, Y)

        # check read actions against expected actions
        expected_reads = np.array([
            [1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 1., 0.]])
        np.testing.assert_array_almost_equal(expected_reads, reads)

    def test_latch(self):
        # test whether our RMM implementation is able to implement a latching
        # switch, i.e. outputting a zero until we have a one at input and
        # then outputting one until the next input one

        # generate input data
        T = 100
        N = 10
        Xs = []
        Ys = []
        for j in range(N):
            # initialize input and output sequence
            X = np.zeros((T, 1))
            Y = np.zeros((T, 1))
            # sample two switching points randomly
            t_1 = random.randrange(10, 40)
            t_2 = random.randrange(60, 90)
            # set input at precisely these locations to one
            X[0] = 1.
            X[t_1] = 1.
            X[t_2] = 1.
            # activate output between these points
            Y[t_1:t_2] = 1.
            # append
            Xs.append(X)
            Ys.append(Y)

        # set up model
        m = 64
        K = 2
        leak = 0.5
        v = 0.9
        controller_initialization = 'identity'
        permit_duplicates = False
        regul = 1E-7
        washout = 0
        input_normalization = False

        model = rmm.RMM(m = m, v = v, leak = leak, K = K, controller_initialization = controller_initialization,
            permit_duplicates = permit_duplicates, 
            regul=regul, washout=washout,
            input_normalization = input_normalization)

        # train the model
        model.fit(Xs, Ys)

        # test the result
        for j in range(N):
            Yhat = model.predict(Xs[j])

            se = np.sum(np.square(Ys[j] - Yhat))
            self.assertTrue(se < 0.1, 'SE was too high, got %g; predicted vs actual:\n%s' % (se, str(np.concatenate([Yhat, Ys[j]], axis=1))))

if __name__ == '__main__':
    unittest.main()
