import numpy as np
import cvxpy as cvx
import scipy as sp

import matplotlib.pyplot as plt

class reservoir():

    def __init__(self, n, phi, radius=1, bias_var=1, A=None, Bw=None, bias=None):

        self.n = int(n)

        self.radius = radius
        self.phi = phi

        # A bunch of hyperparameters
        self.bias_var = bias_var
        if bias is None:
            self.bias = np.random.randn(self.n, 1) * self.bias_var
        else:
            self.bias = bias

        self.A = A
        self.Bw = Bw

    def init_li_ESN(self, alpha, radius):
        # Leaky integrator echo state network
        self.A = (1-alpha) * np.eye(self.n)
        self.Bw = alpha * radius * np.random.randn(self.n, self.n) / np.sqrt(self.n)

    def sr_ESN(self, radius):
        # Bw has spectral radius of radius
        self.A = np.zeros((self.n, self.n))
        Bw = np.random.randn(self.n, self.n) / np.sqrt(self.n)
        sr = np.max(np.abs(np.linalg.eigvals(Bw)))

        self.Bw = Bw * radius / sr

    def linear_ESN(self, radius):
        self.Bw = np.zeros((self.n, self.n))

        A = np.random.randn(self.n, self.n)
        sr = np.max(np.abs(np.linalg.eigvals(A)))
        self.A = A / sr * radius

        # eigs = np.linalg.eigvals(self.A)
        # plt.plot(np.real(eigs), np.imag(eigs), '.')

        # theta = np.linspace(0, 2*np.pi, 100)
        # plt.plot(np.sin(theta), np.cos(theta))
        # plt.show()

    def init_rand(self, A_rad, B_rad):

        A = np.random.randn(self.n, self.n)
        sr = np.max(np.abs(np.linalg.eigvals(A)))
        self.A = A / sr * A_rad

        Bw = np.random.randn(self.n, self.n)
        sr = np.max(np.abs(np.linalg.eigvals(Bw)))
        self.Bw = Bw / sr * B_rad

    def sn_ESN(self, radius):
        self.A = np.zeros((self.n, self.n))
        self.Bw = np.random.randn(self.n, self.n)
        [U, S, V] = np.linalg.svd(self.Bw)

        self.Bw = radius * self.Bw / S[0]


    def project_stable(self, rate=0.999, diag_E=False, diag_P=False, samples=2000, verbose=False):

        solver_tol = 1E-6
        eps = 1E-3

        multis = cvx.Variable((self.n), 'lambdas', nonneg=True)
        T = cvx.diag(multis)

        if diag_E:
            E = cvx.diag(cvx.Variable((self.n, 1), 'E'))
        else:
            E = cvx.Variable((self.n, self.n), 'E')

        if diag_P:
            P = cvx.diag(cvx.Variable((self.n, 1), 'P'))
        else:
            P = cvx.Variable((self.n, self.n), 'P', symmetric=True)

        F = cvx.Variable((self.n, self.n), 'F')
        # F = np.zeros((self.n, self.n))
        W = cvx.Variable((self.n, self.n), 'W')

        Cv = np.eye(self.n)

        # Behavioural description of nonlinearity
        alpha = 0
        beta = 1
        Gamma_v = sp.linalg.block_diag(Cv, np.eye(self.n))
        M = cvx.bmat([[-2 * alpha * beta * T, (alpha + beta) * T], [(alpha + beta) * T, - 2 * T]])

        # Construct final LMI.
        z1 = np.zeros((self.n, self.n))
        z2 = np.zeros((self.n, self.n))

        Mat11 = cvx.bmat([[rate*(E + E.T - P), z1], [z1.T, z2]]) - Gamma_v.T @ M @ Gamma_v
        Mat21 = cvx.bmat([[F, W]])
        Mat22 = P

        Mat = cvx.bmat([[Mat11, Mat21.T], [Mat21, Mat22]])

        # epsilon ensures strict feasability
        constraints = [Mat >> (solver_tol) * np.eye(Mat.shape[0]),
                       P >> (eps + solver_tol) * np.eye(self.n),
                       E + E.T >> (eps + solver_tol) * np.eye(self.n)]

        # stack up xtild, wtild, xtild_next
        # if sample:
        # samples = 5*self.n
        samples = 2000
        xtild = 2 * np.random.randn(self.n, samples)
        xtild_next = self.A @ xtild + self.Bw @ self.phi(xtild + self.bias)
        wtild = self.phi(xtild + self.bias)
        # else:
        #     # Target distribution of eigenvalues.
        #     xtild = np.eye(self.n)
        #     xtild_next = self.A @ xtild + self.Bw @ self.phi(xtild + self.bias)
        #     wtild = self.phi(xtild + self.bias)

        # Empirical covariance matrix
        ztild = np.concatenate([xtild_next, xtild, wtild], 0)
        PHI = ztild @ ztild.T

        # Construct LREE objective
        R = cvx.Variable((2*self.n + self.n, 2*self.n + self.n))
        EFK = cvx.bmat([[E, -F, -W]])
        Q = cvx.bmat([[R, EFK.T], [EFK, E + E.T - np.eye(self.n)]])
        constraints.append(Q >> 0)

        objective = cvx.Minimize(cvx.trace(PHI @ R))

        # solve problem
        prob = cvx.Problem(objective, constraints)
        prob.solve(solver=cvx.SCS, verbose=verbose)

        print("Initilization Status: ", prob.status)
        print("Setup Time: ", prob.solver_stats.setup_time / 1000)
        print("Solve Time: ", prob.solver_stats.solve_time / 1000)

        # Create structure to save
        implicit_reservoir = {"E": E.value, "bias": self.bias,
                              "F": F.value, "W": W.value, "P": P.value,
                              "lambdas": multis.value}

        return implicit_reservoir


