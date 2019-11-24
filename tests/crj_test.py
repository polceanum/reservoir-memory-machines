#!/usr/bin/python3
"""
Tests the Cython implementation of cycle reservoirs with jumps

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

import unittest
import numpy as np
import rmm.crj as crj

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen, Alexander Schulz'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class TestCRJ(unittest.TestCase):

    def test_setup_input_weight_matrix(self):
        # test for a small case with manually set signs
        v = 0.7
        U_expected = np.array([[-v, -v, -v, -v], [+v, +v, -v, +v]])
        U_actual = crj.setup_input_weight_matrix(U_expected.shape[1], U_expected.shape[0], v)
        np.testing.assert_array_almost_equal(U_expected, U_actual)

        # test with shifted start
        U_expected = np.array([[-v, -v, -v, +v], [+v, -v, +v, +v]])
        U_actual = crj.setup_input_weight_matrix(U_expected.shape[1], U_expected.shape[0], v, 1)
        np.testing.assert_array_almost_equal(U_expected, U_actual)

    def test_setup_reservoir_matrix(self):

        w_c = 0.7
        w_j = 0.9
        l   = 2

        # test an example where m is divisble by l
        W_expected = np.array([
            [0., 0., w_j, w_c],
            [w_c, 0., 0., 0.],
            [w_j, w_c, 0., 0.],
            [0., 0., w_c, 0.]])
        W_actual = crj.setup_reservoir_matrix(W_expected.shape[0], w_c, w_j, l)
        np.testing.assert_array_almost_equal(W_expected, W_actual.toarray())


        # test an example where that is not the case
        W_expected = np.array([
            [0., 0., w_j, 0., w_c],
            [w_c, 0., 0., 0., 0.],
            [w_j, w_c, 0., 0., w_j],
            [0., 0., w_c, 0., 0.],
            [0., 0., w_j, w_c, 0.]])
        W_actual = crj.setup_reservoir_matrix(W_expected.shape[0], w_c, w_j, l)
        np.testing.assert_array_almost_equal(W_expected, W_actual.toarray())

if __name__ == '__main__':
    unittest.main()
