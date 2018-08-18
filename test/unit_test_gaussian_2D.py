import unittest

import numpy as np
import rhmm
import os

from scipy.optimize import linear_sum_assignment

def classification_mat(seqs1, seqs2):
    seqs1_flat = np.array([item1 for seq1 in seqs1 for item1 in seq1])
    seqs2_flat = np.array([item2 for seq2 in seqs2 for item2 in seq2])

    M = max(np.max(seqs1_flat), np.max(seqs1_flat)) + 1

    W = np.empty((M, M), dtype=int)
    for i in range(M):
        W[i] = np.bincount(seqs2_flat[seqs1_flat == i], minlength=M)
    return W

def find_best_mapping(classification_mat):
    return linear_sum_assignment(-classification_mat)

# def accuracy(classification_mat):
#     i1, i2 = find_best_mapping(classification_mat)
#     classification_mat = classification_mat[:, i2]
#     return np.sum(np.diag(classification_mat))/ np.sum(classification_mat)

def HMM_generate_sequences(init_probs, transitions, means, covs, count):

    M = len(init_probs)

    all_points = []
    all_states = []
    for _ in range(count):
        N = np.random.randint(10, 30)
        state = np.random.choice(M, p=init_probs)

        points = []
        states = []

        for _ in range(N):
            p = np.random.multivariate_normal(means[state], covs[state])
            points.append(p)
            states.append(state)
            state = np.random.choice(M, p=transitions[state])

        points = np.array(points)
        states = np.array(states)

        all_points.append(points)
        all_states.append(states)

    return all_points, all_states


def hide_labels(labels, p):

    labels_subsample = []
    for label_sequence in labels:
        hide = np.random.uniform(0, 1, len(label_sequence)) < p
        labels_subsample.append(np.where(hide, -1, label_sequence))
    return labels_subsample


class Gaussian2DHMM(unittest.TestCase):

    def setUp(self):
        self.means = np.array([[ 3.80070008, -3.79715011],
                               [ 3.97795298,  3.77378094],
                               [-3.06194507, -3.53452570],
                               [-2.03436517,  4.17258804]])

        self.covs = np.array([[[  0.9212154 ,   0.05736702],
                               [  0.05736702,   1.86616798]],

                         [[  0.21034409,   0.29034688],
                          [  0.29034688,  12.23792899]],

                         [[  6.24148907,   6.0502486 ],
                          [  6.0502486 ,   6.18252548]],

                         [[  2.90443324,   0.20656606],
                          [  0.20656606,   2.7561749 ]]])


        self.transitions = np.array([[0.7, 0.1, 0.1, 0.1],
                                     [0.2, 0.5, 0.3, 0.0],
                                     [0.3, 0.3, 0.2, 0.2],
                                     [0.05, 0.1, 0.05, 0.8]])

        self.init_probs = np.array([0.3, 0.5, 0.1, 0.1])

        np.random.seed(123)

        self.sequences, self.labels = HMM_generate_sequences(self.init_probs, self.transitions,
                                                             self.means, self.covs, 256)

        self.help_labels = hide_labels(self.labels, 0.8)

        self.hmm = rhmm.HMM()
        for mean, cov in zip(self.means, self.covs):
            self.hmm.add_state(rhmm.MultivariateGaussian(mean, cov))

        self.hmm.fit(self.sequences, self.help_labels, max_iters=100)

        print(np.exp(self.hmm.init_probs))
        print(np.exp(self.hmm.transitions))

    def test_viterbi_acc(self):

        predicted = self.hmm.viterbi(self.sequences, self.help_labels)
        W = classification_mat(predicted, self.labels)
        _, ids = find_best_mapping(W)

        W = W[:, ids]
        transitions = np.exp(self.hmm.transitions[:, ids])
        init_probs = np.exp(self.hmm.init_probs[ids])

        print(transitions)
        print(init_probs)

        acc = np.sum(np.diag(W))/ np.sum(W)
        print("acc is %f" % acc)
        self.assertGreaterEqual(acc, 0.9832)

        norm_transitions = np.linalg.norm(self.transitions - transitions, 1)
        print("transition l1 norm is %f" % norm_transitions)
        self.assertLessEqual(norm_transitions, 0.04387)

        norm_init_probs = np.linalg.norm(self.init_probs - init_probs, 1)
        print("init_probs l1 norm is %f" % norm_init_probs)
        self.assertLessEqual(norm_init_probs, 0.05419)


if __name__ == '__main__':
    unittest.main()
