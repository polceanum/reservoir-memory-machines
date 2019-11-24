"""
Implements reservoir memory machines.
This architecture takes concept from the Neural Turing Machine
developed by Graves, Wayne, and Danihelka (2014). In contrast
to the Neural Turing Machine, this model is not learned via
gradient descent but instead driven by an echo state network
(Jaeger and Maas, 2004), more precisely a cycle reservoir with
jumps (Rodan and Tino, 2012).

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
import numpy as np
from scipy.spatial.distance import cdist
import rmm.crj as crj
import rmm.esn as esn

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen, Alexander Schulz'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class RMM(esn.ESN):
    """ Implements a reservoir memory machine where we write and
    read with a single head in a location-based linear memory.
    The location of the write and read head is controlled by the reservoir.
    The precise controller for writing is trained using alignment and linear
    regression.

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
    K:   The number of memory entries. 16 per default.
    controller_initialization: The strategy for determining the initial write and read head
         controller.
         Can be either 'forward'for a policy that always moves one
         Step ahead or 'identity' (default) for starting with an identity mapping between
         input and output and determining the policy from that point.
    permit_duplicates: Wether the write head should permit duplicates in memory
         if it helps to maintain linearity in the memory structure. False per
         default.
    regul: The L2 regularization strength for linear regression. 1E-5 per default.
    input_normalization: Switch to true if the input should be z-normalized
         before feeding it into the network. True per default.
    washout: The number of washout steps before training. 0 per default.
    _U:  The n x self.m input-to-reservoir matrix. Is set during training.
    _W:  The self.m x self.m recurrent connection matrix as a scipy sparse
         csr matrix. Is set during training.
    _V:  The (trained) L x (self.m + n) matrix mapping from reservoir
         activations and the current memory read to the output.
    _V_w: The (trained) 1 x (n + self.m) write controller matrix mapping from
         the current input and the current reservoir activation to the write
         action. In particular, if _V_w * [X[t, :], H[t, :]] is positive,
         the write head writes the current input to memory. Otherwise, it
         does nothing.
    _V_r: The (trained) 3 x (n + self.m) read controller matrix mapping from
         the current input and the current reservoir activation to the read
         action. In particular, if the first entry of
         _V_r * [X[t, :], H[t, :]] is largest, the read head reads from the
         current memory location and does not move. If the second entry is
         largest, the read head moves one step ahead in memory and then reads.
         If the third entry is largest, the read moves to the beginning of the
         memory and then reads.
    _mu: The mean of the training data for each feature. This is used for
         normalization purposes.
    _beta: The precision of the training data for each feature.
         This is used for normalization purposes.
    """
    def __init__(self, m = 128, v = .1, w_c = 0.9, w_j = 0.3, l = None, leak = 1.,
        K = 16, controller_initialization = 'identity', permit_duplicates = False,
        regul = 1E-5, input_normalization = True, washout = 0 ):
        super(RMM, self).__init__(m, v, w_c, w_j, l, leak, regul, input_normalization, washout)
        self.K = K
        self.controller_initialization = controller_initialization
        self.permit_duplicates = permit_duplicates

    def _apply_write(self, X, H):
        """ Applies only the write head to the input time series and generates
        an according memory trace.

        Args:
        X: The T x n input time series.
        H: The T x self.m time series of ESN reservoir activations for the
           input.

        Returns:
        Ms: A T x self.K x n tensor of memory states at each point in time.
        """
        T = len(X)
        if len(X) != len(H):
            raise ValueError('Internal error: The lengths of the input and reservoir activation time series do not match! (%d vs %d)' % (len(X), len(H)))
        XH = np.concatenate([X, H, np.ones((len(X), 1))], axis=1)
        # compute the predicted write head actions
        write_actions = np.dot(XH, self._V_w.T)

        # initialize the memory trace
        Ms = np.zeros((T, self.K, X.shape[1]))
        # start computing the writes
        write_loc = 0
        for t in range(T):
            # copy the memory from the last time step
            if t > 0:
                Ms[t, :, :] = Ms[t-1, :, :]
            # check the write action
            if write_actions[t] > 0.:
                # write into memory
                Ms[t, write_loc, :] = X[t, :]
                write_loc += 1
                if write_loc >= self.K:
                    write_loc = 0
        # and return
        return Ms

    def _apply_read(self, X, H):
        """ Moves the write head over the input time series and the read head
        over the memory, as determined by the matrices self._V_w and self._V_r
        respectively. This results in a time series of reads from the memory
        which can be used to support the output regression of the RMM.

        Args:
        X: The T x n input time series.
        H: The T x self.m time series of ESN reservoir activations for the
           input.

        Returns:
        reads: A T x n time series of memory reads.
        """
        T = len(X)
        if len(X) != len(H):
            raise ValueError('Internal error: The lengths of the input and reservoir activation time series do not match! (%d vs %d)' % (len(X), len(H)))
        XH = np.concatenate([X, H, np.ones((len(X), 1))], axis=1)
        # compute the predicted write head actions
        write_actions = np.dot(XH, self._V_w.T)
        # compute the predicted read head actions
        read_actions = np.dot(XH, self._V_r.T)
        read_actions = np.argmax(read_actions, axis=1)

        # initialize the memory
        M = np.zeros((self.K, X.shape[1]))
        # start computing the reads
        write_loc = 0
        read_loc  = 0
        reads = []
        for t in range(T):
            # check the write actions first
            if write_actions[t] > 0.:
                # write into memory
                M[write_loc, :] = X[t, :]
                write_loc += 1
                if write_loc >= self.K:
                    write_loc = 0
            # then check the read action
            if read_actions[t] == 0:
                # do nothing
                pass
            elif read_actions[t] == 1:
                # do one step
                read_loc += 1
                if read_loc >= self.K:
                    read_loc = 0
            elif read_actions[t] == 2:
                # reset
                read_loc = 0
            else:
                raise ValueError('Internal error; unexpected read action: %d' % read_actions[t])
            # read from memory
            reads.append(np.copy(M[read_loc, :]))
        # concatenate reads
        reads = np.stack(reads, axis=0)
        # and return
        return reads

    def _apply_reservoir(self, X):
        """ Applies this RMM to the given input time series, i.e. applies both
        a standard ESN and the write/read head mechanism, which is driven by
        the ESN.

        Args:
        X: The T x n input time series.

        Returns:
        HR: A T x (self.m + n) time series of reservoir activations,
            concatenated with memory reads.
        """
        T = X.shape[0]
        n = X.shape[1]
        if n != self._U.shape[1]:
            raise ValueError('The number of input dimensions does not match the training data. Expected %d dimension but got %d dimensions!' % (self._U.shape[1], n))
        if self.K < 1:
            raise ValueError('The memory size must be at least 1')

        # apply the standard ESN reservoir
        H = super(RMM, self)._apply_reservoir(X)

        # apply the write and read head
        reads = self._apply_read(X, H)

        # concatenate both and return
        return np.concatenate([H, reads], axis=1)

    def _align_write(self, X, Y):
        """ Aligns the given input sequence to the given output sequence,
        based on the current input-to-output mapping self._V.

        In more detail, this method finds for each output element the
        closest input element after multiplying with self._V. These are
        then the elements we try to write to memory.

        Args:
        X: A T x n input time series.
        Y: A T x L output time series.

        Returns:
        write_actions: A time series with T entries, either -1, if no
            write should be applied at time t, or +1, if the write head
            should write to memory at that time.
        """
        if len(X) != len(Y):
            raise ValueError('The lengths of the input and output time series do not match! (%d vs %d)' % (len(X), len(Y)))
        T = len(Y)
        # compute all pairwise costs that can occur between input and output
        Delta = cdist(np.dot(X, self._V.T), Y)

        # get the lowest distance between vectors in X and in Y.
        # The vectors in X corresponding to these lowest distances
        # are what we wish to store in memory
        delta_min = np.min(Delta, axis=0)

        # infer the necessary memory writes via backtracing
        writes = set()
        if self.permit_duplicates:
            # perform backtracing and infer the optimal memory reads
            i = 0
            j = 0
            duplicates = set()
            read_actions  = []
            while j < T:
                if Delta[i, j] < delta_min[j] + 1E-3:
                    # if we have already written this time step to memory
                    # and the next time step would also be fine,
                    # write the next step instead
                    if i < T-1 and i in writes and Delta[i+1,j] < delta_min[j] + 1E-3:
                        writes.add(i+1)
                        i += 1
                        j += 1
                    else:
                        # aligning is co-optimal, so we perform a write here
                        writes.add(i)
                        j += 1
                else:
                    # otherwise we increment i
                    i += 1
                    # and we wrap around at T
                    if i >= T:
                        i = 0
        else:
            # if we do not permit duplicates, try to minimize the amount
            # of memory writes by always checking whether an element that
            # we have already written would be co-optimal. In other words,
            # we always take the smallest index that is also minimal
            for j in range(T):
                i = np.where(Delta[:, j] < np.min(Delta[:, j]) + 1E-3)[0][0]
                writes.add(i)

        if len(writes) > self.K:
            raise ValueError('We need at least %d writes but memory has only size %d' % (len(writes), self.K))

        # based on the memory writes we can infer the optimal write head
        # actions
        write_loc = 0
        write_actions = np.zeros(T)
        for t in range(T):
            if t in writes:
                write_loc +=1
                if write_loc >= self.K:
                    write_loc = 0
                write_actions[t] = 1.
            else:
                write_actions[t] = -1.
        return write_actions

    def _align_read(self, Ms, Y):
        """ Generates an optimal alignment between the given memory trace and
        the given output time series, based on the current projection matrix
        self._V.

        In more detail, let k be the current location in memory and let t
        be the current time step in the output time series. Then, we obtain
        the following dynamic programming relation:

        D[k, t] = ||Ms[t, k, :] - Y[t, :]|| + min(
            D[k, t+1], for a pause action
            D[k+1, t+1], for a step action
            D[0, t+1]) for a reset action

        The base case is the last column, i.e. D[k, T], which is given as
        D[k, T] = ||Ms[T, k, :] - Y[T, :]||

        We can obtain the optimal action sequence via backtracing.

        Args:
        Ms: A T x self.K x n tensor of memory states at each point in
            time
        Y:  A T x L output time series.

        Returns:
        cost: The minimal cost of aligning the memory trace with the
              output time series.
        read_actions: A T x 3 time series with entries [1., 0. 0.] if
              the read head should not move at time t, [0., 1., 0.] if
              the read head should move one step ahead at time t, or
              [0., 0., 1.] if the read head should jump back to the
              start of the memory at time t.
        """
        T = len(Y)
        # compute all pairwise costs that can occur
        Delta = np.zeros((self.K, T))
        for k in range(self.K):
            for t in range(T):
                Delta[k, t] = np.linalg.norm(np.dot(self._V, Ms[t, k, :]) - Y[t, :])

        # initialize dynamic programming matrix
        D = np.zeros((self.K, T))
        # set up the base case in the last column
        for k in range(self.K):
            D[k, T-1] = Delta[k, T-1]
        # perform dynamic programming
        for t in range(T-2, -1, -1):
            for k in range(self.K-1):
                D[k, t] = Delta[k, t] + min(D[k, t+1], min(D[k+1,t+1], D[0,t+1]))
            # for the last memory entry, reset and step are the same actions,
            # such that we need a different recurrence
            D[self.K-1, t] = Delta[self.K-1, t] + min(D[self.K-1, t+1], D[0,t+1])

        # perform backtracing to find the optimal actions
        actions = []
        # first, determine the starting position
        if D[0, 0] < D[1, 0] + 1E-3:
            k = 0
            d = D[0, 0]
            actions.append(np.array([1., 0., 0.]))
        else:
            k = 1
            d = D[1, 0]
            actions.append(np.array([0., 1., 0.]))
        # then, iterate over all points in time and infer the optimal action
        for t in range(T-1):
            # check if pause is optimal
            if Delta[k, t] + D[k,t+1] < D[k, t] + 1E-3:
                # if so, add a pause action
                actions.append(np.array([1., 0., 0.]))
            # check if step is optimal
            elif k < self.K-1 and Delta[k, t] + D[k+1, t+1] < D[k, t] + 1E-3:
                # if so, add a step action
                actions.append(np.array([0., 1., 0.]))
                k += 1
            # check if reset is optimal
            elif Delta[k, t] + D[0, t+1] < D[k, t] + 1E-3:
                # if so, add a reset action (or a step action, if we are
                # at the end of memory anyways)
                if k == self.K-1:
                    actions.append(np.array([0., 1., 0.]))
                else:
                    actions.append(np.array([0., 0., 1.]))
                k = 0
            else:
                # if neither applies there is an error
                raise ValueError('internal error: There was no optimal option in dynamic programming, which can not happen.')

        # return the result of the dynamic programming scheme
        return min(D[0, 0], D[1, 0]), actions

    def fit(self, X, Y):
        """ Fits this RMM to the given input and output time series.
        In more detail, this method sets the properties self._mu and
        self._beta for normalization, self._V_w for controlling the write
        head, self._V_r for controlling the read head, and self._V for
        the final mapping from reservoir activations and memory traces to the
        output. Because the V matrices are dependend on each other (e.g. the
        movement of the write head determined what memory can be read by the
        read head, and the movement of the read head determined what can be
        mapped to the output) we apply an alternating optimization scheme to
        map the input to the output.

        Args:
        X: A T x n input time series.
        Y: A T x L output time series.

        Returns:
        self: This object.
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

        # normalize all inputs
        for j in range(len(Xs)):
            Xs[j] = (Xs[j] - self._mu) * self._beta

        # prepate the output
        if isinstance(Y, list):
            Ys = Y
        else:
            Ys = [Y]
        for j in range(len(Ys)):
            # remove the washout steps from each output matrix
            Ys[j] = Ys[j][self.washout:, :]
        Y = np.concatenate(Ys, axis=0)

        # initialize the reservoir
        self._init_reservoir(n)

        # process all input matrices with a cycle reservoir with jumps
        # and concatenate them to one big output
        Hs = []
        XHs = []
        for X in Xs:
            H = super(RMM, self)._apply_reservoir(X)
            # remove washout steps before recording the data
            Hs.append(H[self.washout:, :])
            XHs.append(np.concatenate([X[self.washout:, :], H[self.washout:, :], np.ones((len(X)-self.washout, 1))], axis=1))
        XH = np.concatenate(XHs, axis=0)

        # pre-compute the pseudo-inverse based of XH in order to
        # save computations in the alternating optimization
        # later on
        C    = np.dot(XH.T, XH)
        Cinv = np.linalg.inv(C + self.regul * np.eye(XH.shape[1]))
        XH_pinv = np.dot(XH, Cinv)

        # initialize the controller
        if self.controller_initialization == 'forward':
            # initialize the controller to always move both the write and the read
            # head forward
            write_actions = np.ones((len(XH), 1))
            read_actions  = np.zeros((len(XH), 3))
            read_actions[:, 1] = 1.

            # apply linear regression to infer the correct matrices
            # for controlling both the write head
            self._V_w = np.dot(write_actions.T, XH_pinv)
            # and the read read
            self._V_w = np.dot(read_actions.T, XH_pinv)
        elif self.controller_initialization == 'identity':
            # initialize the mapping from memory to output as identity
            self._V = np.eye(Y.shape[1], n)
            # determine optimal write actions via alignment
            write_actions = []
            for j in range(len(Xs)):
                # determine optimal write actions first
                write_actions_j = self._align_write(Xs[j][self.washout:, :], Ys[j][self.washout:, :])
                write_actions.append(write_actions_j)
            write_actions = np.expand_dims(np.concatenate(write_actions), 1)
            # apply linear regression to infer the best possible write controller
            self._V_w = np.dot(write_actions.T, XH_pinv)
            # then optimize the read controller
            read_actions  = []
            for j in range(len(Xs)):
                # apply the write controller to infer the actual writes
                Ms = self._apply_write(Xs[j][self.washout:, :], Hs[j][self.washout:, :])
                # apply alignment to infer the optimal read actions
                _, read_actions_j = self._align_read(Ms, Ys[j][self.washout:, :])
                read_actions.append(read_actions_j)
            read_actions = np.concatenate(read_actions, axis=0)
            # and the read read
            self._V_r = np.dot(read_actions.T, XH_pinv)
        else:
            raise ValueError('Unrecognized controller initialization method: %s' % self.controller_initialization)

        # now we go into our alternating optimization scheme.
        last_loss = np.inf
        last_Vs = None
        for epoch in range(30):
            # First step: Compute the current reads based on the
            # initial write and read head controller

            Reads = []
            for j in range(len(Xs)):
                read = self._apply_read(Xs[j][self.washout:, :], Hs[j][self.washout:, :])
                Reads.append(read)
            Reads = np.concatenate(Reads, axis=0)

            # next, apply linear regression to map the reads to
            # the output matrix
            self._V = esn.linreg(Reads, Y, self.regul)
            # compute the current loss
            loss = np.sqrt(np.mean(np.sum(np.square(Y - np.dot(Reads, self._V.T)), axis=1)))
            # check if we got better
            if loss < last_loss - 1E-3:
                last_loss = loss
                last_Vs   = (self._V_w, self._V_r, self._V)
            else:
                # if not, break off the search and revert the last step
                self._V, self._V_r, self._V = last_Vs
                break

            # next, optimize the write head via alignment
            write_actions = []
            for j in range(len(Xs)):
                # determine optimal write actions first
                write_actions_j = self._align_write(Xs[j][self.washout:, :], Ys[j][self.washout:, :])
                write_actions.append(write_actions_j)
            write_actions = np.expand_dims(np.concatenate(write_actions), 1)
            # apply linear regression to infer the best possible write controller
            self._V_w = np.dot(write_actions.T, XH_pinv)
            # then optimize the read controller
            read_actions  = []
            for j in range(len(Xs)):
                # apply the write controller to infer the actual writes
                Ms = self._apply_write(Xs[j][self.washout:, :], Hs[j][self.washout:, :])
                # apply alignment to infer the optimal read actions
                _, read_actions_j = self._align_read(Ms, Ys[j][self.washout:, :])
                read_actions.append(read_actions_j)
            read_actions = np.concatenate(read_actions, axis=0)
            # and the read read
            self._V_r = np.dot(read_actions.T, XH_pinv)

        # after completing the alternating optimization scheme, apply the
        # reads with the current policy once more and concatenate the reads
        # to the reservoir activations
        HR = []
        for j in range(len(Xs)):
            read = self._apply_read(Xs[j][self.washout:, :], Hs[j][self.washout:, :])
            HR.append(np.concatenate([Hs[j], read], axis=1))
        HR = np.concatenate(HR, axis=0)
        # apply linear regression to map reservoir and reads to the output
        self._V = esn.linreg(HR, Y, self.regul)

        # return model
        return self
