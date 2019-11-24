#!/usr/bin/python3
"""
Tests the our python implementation of echo state networks.

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

import random
import unittest
import time
import numpy as np
import rmm.esn as esn

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen, Alexander Schulz'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class TestESN(unittest.TestCase):

    def test_addition(self):
        # test whether our ESN implementation is able to add two input
        # signals, which needs no memory at all and should thus be doable
        # with cycle weight and jump weight zero

        # generate training data
        T = 500
        X = np.random.randn(T, 2)
        Y = np.expand_dims(X[:, 0] + X[:, 1], 1)

        # set hyper-parameters for the ESN
        m = 256
        v = 0.1
        w_c = 0.
        w_j = 0.
        regul = 1E-7

        model = esn.ESN(m = m, v = v, w_c = w_c, w_j = w_j, regul=regul)

        # train the model
        model.fit(X, Y)

        # test the result
        Yhat = model.predict(X)

        mse = np.mean(np.square(Y - Yhat))
        self.assertTrue(mse < 0.1, 'MSE was too high, got %g' % mse)

    def test_sine_to_cosine(self):
        # test whether our ESN implementations are able to transform an
        # input sine wave to an output cosine wave. This does require
        # memory, but not much and is still fairly simple

        # generate training data
        T = 300
        N = 3
        Xs = []
        Ys = []
        for j in range(N):
            ts = random.random() + np.linspace(0,1,T)
            X = np.expand_dims(np.sin(ts * 4 * np.pi), 1)
            Y = np.expand_dims(np.cos(ts * 4 * np.pi), 1)
            Xs.append(X)
            Ys.append(Y)

        # set hyper-parameters for the ESN
        m = 64
        v = 0.1
        w_c = 0.5
        w_j = 0.3
        regul = 1E-7
        washout = 10

        models = [
            esn.ESN(m = m, v = v, w_c = w_c, w_j = w_j, regul=regul, washout=washout),
            esn.ESGRU(m = m, v = v, w_c = w_c, w_j = w_j, regul=regul, washout=washout)
        ]

        for model in models:

            # train the model
            model.fit(Xs, Ys)

            # test the result
            for j in range(N):
                Yhat = model.predict(Xs[j])

                mse = np.mean(np.square(Ys[j] - Yhat))
                self.assertTrue(mse < 0.1, 'MSE was too high, got %g' % mse)

    def test_delayed_response(self):
        # test whether our ESN implementation is able to respond to an
        # input pulse with a delayed response pulse. This requires memory
        # which scales linearly with the delay

        # generate input data
        T = 300
        delay = 100
        N = 5
        Xs = []
        Ys = []
        for j in range(N):
            t_offset = random.randrange(20, delay)
            X = np.zeros((T, 1))
            X[t_offset, :] = 1.
            Y = np.zeros((T, 1))
            Y[t_offset + delay, :] = 1.
            Xs.append(X)
            Ys.append(Y)

        # set hyper-parameters for the ESN
        m = 128
        v = 0.1
        w_c = 0.99
        w_j = 0.
        regul = 1E-7
        washout = 0
        input_normalization = False

        model = esn.ESN(m = m, v = v, w_c = w_c, w_j = w_j, regul=regul, washout=washout, input_normalization = input_normalization)

        # train the model
        model.fit(Xs, Ys)

        # test the result
        for j in range(N):
            Yhat = model.predict(Xs[j])
            se = np.sum(np.square(Ys[j] - Yhat))
            self.assertTrue(se < 0.1, 'SE was too high, got %g' % se)

        # if we reduce the number of neurons below the delay, we expect that
        # this does not work anymore
        m = delay - 1
        model = esn.ESN(m = m, v = v, w_c = w_c, w_j = w_j, regul=regul, washout=washout, input_normalization = input_normalization)

        # train the model
        model.fit(Xs, Ys)

        # test the result
        for j in range(N):
            Yhat = model.predict(Xs[j])
            se = np.sum(np.square(Ys[j] - Yhat))
            self.assertTrue(se > 0.5, 'SE was unexpectedly low, got %g' % se)


        # and setting the leak rate to small values should not help either
        leak = 0.01
        model = esn.ESN(m = m, v = v, w_c = w_c, w_j = w_j, leak = leak, regul=regul, washout=washout, input_normalization = input_normalization)

        # train the model
        model.fit(Xs, Ys)

        # test the result
        for j in range(N):
            Yhat = model.predict(Xs[j])
            se = np.sum(np.square(Ys[j] - Yhat))
            self.assertTrue(se > 0.5, 'SE was unexpectedly low, got %g' % se)

if __name__ == '__main__':
    unittest.main()
