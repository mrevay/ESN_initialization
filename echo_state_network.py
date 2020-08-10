import numpy as np
import scipy as sp
import cvxpy as cvx
import matplotlib.pyplot as plt
import datasets.loader as loader

from hyperopt import tpe
from hyperopt import hp
from hyperopt import fmin
from hyperopt import Trials


class echo_state_network:

    def __init__(self, n, m, p, q, connectivity, radius, phi, washout, bias_var=1, Bu_var=1, Du_var=1, ofb=True):

        self.n = int(n)
        self.m = int(m)
        self.p = int(p)
        self.q = int(q)

        self.washout = washout
        self.connectivity = connectivity

        self.radius = radius
        self.phi = phi

        self.alpha = 0
        self.beta = 1

        # A bunch of hyperparameters
        self.bias_var = bias_var
        self.Bu_var = Bu_var
        self.Du_var = Du_var

        self.ofb = ofb


    def ESN_init_orthog(self):

        self.A = np.zeros((self.n, self.n))

        # W = sp.sparse.random(self.n, self.n, self.connectivity) / np.sqrt(self.n)
        W = np.random.randn(self.n, self.n) / np.sqrt(self.n)
        [U, S, V] = np.linalg.svd(W)

        self.Bw = self.radius * U @ V

        self.bias = self.bias_var * np.random.randn(self.q, 1)
        self.Bu = self.Bu_var * np.random.randn(self.n, self.m)
        self.Du = self.Du_var * np.random.randn(self.q, self.m)
        self.Cv = np.eye(self.n)

        if self.ofb:
            self.Wofb = np.random.randn(self.n, self.p) / np.sqrt(self.p)

    def ESN_init_sr(self):

        self.A = np.zeros((self.n, self.n))

        W = sp.sparse.random(self.n, self.n, self.connectivity) / np.sqrt(self.n)
        # W = np.random.randn(self.n, self.n) / np.sqrt(self.n)

        eig = sp.sparse.linalg.eigs(W, 1)
        self.Bw = self.radius * W / np.abs(eig[0])

        self.bias = self.bias_var * np.random.randn(self.q, 1)
        self.Bu = self.Bu_var * np.random.randn(self.n, self.m)
        self.Du = self.Du_var * np.random.randn(self.q, self.m)
        self.Cv = np.eye(self.n)

        # Add output feedback weight if necessary
        if self.ofb:
            self.Wofb = np.random.randn(self.n, self.p) / np.sqrt(self.p)

    def ESN_init_IEE(self, diag_E=False, diag_P=False):

        solver_tol = 1E-6
        eps = 1E-3

        multis = cvx.Variable((self.q), 'lambdas', nonneg=True)
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
        # F = self.radius*np.random.randn(self.n, self.n) / np.sqrt(self.n)

        # Bw = cvx.Variable((self.n, self.q), 'Bw')
        Bw = np.zeros((self.n, self.n))

        # Cv = np.random.randn(self.q, self.n) / np.sqrt(self.n)
        # Cv = np.random.randn(self.n, self.n) / np.sqrt(self.n) + 1E-1 * np.eye(self.n)
        Cv = np.eye(self.n)

        Gamma_v = sp.linalg.block_diag(Cv, np.eye(self.q))

        M = cvx.bmat([[-2 * self.alpha * self.beta * T, (self.alpha + self.beta) * T], [(self.alpha + self.beta) * T, - 2 * T]])

        # Construct final LMI.
        z1 = np.zeros((self.n, self.q))
        z2 = np.zeros((self.q, self.q))

        Mat11 = cvx.bmat([[E + E.T - P, z1], [z1.T, z2]]) - Gamma_v.T @ M @ Gamma_v
        Mat21 = cvx.bmat([[F, Bw]])
        Mat22 = P

        Mat = cvx.bmat([[Mat11, Mat21.T], [Mat21, Mat22]])

        # epsilon ensures strict feasability
        constraints = [Mat >> (solver_tol) * np.eye(Mat.shape[0]),
                       P >> (eps + solver_tol) * np.eye(self.n),
                       E + E.T >> (eps + solver_tol) * np.eye(self.n)]

        # Target distribution of eigenvalues
        Ass = self.radius * np.random.randn(self.n, self.n,) / np.sqrt(self.n)
        self.Atild = Ass

        # objective = cvx.Minimize(cvx.norm(E @ Ass - F) + cvx.norm(Bw) + cvx.norm(Dw))
        objective = cvx.Minimize(cvx.norm(E @ np.linalg.pinv(Cv) @ Ass - Bw - F))
        # objective = cvx.Minimize(cvx.norm(E @ Ass - Bw @ Cv - F))
        # objective = cvx.Minimize(cvx.norm(E @ Ass - Bw))

        prob = cvx.Problem(objective, constraints)
        prob.solve(solver=cvx.SCS, verbose=True)
        print("Initilization Status: ", prob.status)
        print("Setup Time: ", prob.solver_stats.setup_time / 1000)
        print("Solve Time: ", prob.solver_stats.solve_time / 1000)

        # Initialize reservoir
        Einv = np.linalg.inv(E.value)

        if isinstance(F, np.ndarray):
            self.A = Einv @ F
        else:
            self.A = Einv @ F.value

        if isinstance(Bw, np.ndarray):
            self.Bw = Einv @ Bw
        else:
            self.Bw = Einv @ Bw.value

        # self.Bw = Einv @ Bw.value
        # self.Bw = Einv @ Bw

        self.bias = self.bias_var * np.random.randn(self.q, 1)
        self.Bu = self.Bu_var * np.random.randn(self.n, self.m)
        self.Du = self.Du_var * np.random.randn(self.q, self.m)
        self.Cv = Cv

        self.P = P.value
        self.E = E.value
        self.multis = multis.value

        # Add output feedback weight if necessary
        if self.ofb:
            self.Wofb = np.random.randn(self.n, self.p) / np.sqrt(self.p)

    def train(self, data):

        u = data["u"]
        y = data["y"]

        #Simulate ESN dynamics on training data
        X = np.zeros((self.n, y.shape[1]))
        for t in range(1, y.shape[1]):
            vt = self.Cv @ X[:, t-1:t] + self.Du @ u[:, t:t+1] + self.bias
            wt = self.phi(vt)

            if self.ofb:
                X[:, t:t+1] = self.A @ X[:, t-1:t] + self.Bw @ wt + \
                              self.Bu @ u[:, t:t+1] + self.Wofb @ y[:, t-1:t]
            else:
                X[:, t:t+1] = self.A @ X[:, t-1:t] + self.Bw @ wt + self.Bu @ u[:, t:t+1]

        Ytilde = y[:, self.washout:]
        Xtilde = X[:, self.washout:]
        self.Cy = Ytilde @ np.linalg.pinv(Xtilde)

    def test(self, data):

        u = data["u"]
        y = data["y"]

        #Simulate ESN dynamics on training data
        X = np.zeros((self.n, y.shape[1]))
        for t in range(1, y.shape[1]):
            vt = self.Cv @ X[:, t-1:t] + self.Du @ u[:, t:t+1] + self.bias
            wt = self.phi(vt)
            if self.ofb:
                if t < self.washout:
                    X[:, t:t+1] = self.A @ X[:, t-1:t] + self.Bw @ wt + \
                                  self.Bu @ u[:, t:t+1] + self.Wofb @ y[:, t-1:t]
                else:
                    X[:, t:t+1] = (self.Wofb @ self.Cy + self.A) @ X[:, t-1:t] +\
                                   self.Bw @ wt + self.Bu @ u[:, t:t+1]

            else:
                X[:, t:t+1] = self.A @ X[:, t-1:t] + self.Bw @ wt + self.Bu @ u[:, t:t+1]

        Ytilde = y[:, self.washout:]
        Xtilde = X[:, self.washout:]

        Yhat = self.Cy @ Xtilde

        perf = {}
        perf["NRMSE"] = np.linalg.norm(Ytilde - Yhat) / np.linalg.norm(Ytilde)

        return np.linalg.norm(Ytilde - Yhat) / np.linalg.norm(Ytilde)


if __name__ == "__main__":

    param_search = False

    def phi(x):
        # return np.maximum(x, 0)
        return np.tanh(x)
        # return x

    # [train, val, test] = loader.load_data(dataset='WH')
    [train, val, test] = loader.load_data(dataset='Silverbox')

    if param_search:
        def train_and_val(hyperparameters):

            n = hyperparameters["n"]
            radius = hyperparameters["radius"]
            Bu_var = hyperparameters["Bu_var"]
            Du_var = hyperparameters["Du_var"]
            bias_var = hyperparameters["bias_var"]
            density = hyperparameters["density"]

            ESN = echo_state_network(n, 1, 1, n, density, radius, phi, 200,
                                    bias_var=bias_var, Du_var=Du_var,
                                    Bu_var=Bu_var)
            # ESN.ESN_Init_IEE()
            ESN.ESN_init_sr()
            ESN.train(train)

            return ESN.test(val)

        param_space = {"n": hp.quniform("n", 100, 250, 5),
                    "radius": hp.uniform("radius", 0.5, 1.0),
                    "Bu_var": hp.uniform("Bu_var", 0.1, 4.5),
                    "Du_var": hp.uniform("Du_var", 0.1, 2.5),
                    "bias_var": hp.uniform("bias_var", 0.1, 5.0),
                    "density": hp.uniform("density", 0.2, 0.99)}

        MAX_EVALS = 500
        bayes_trials = Trials()

        # Optimize
        best = fmin(fn=train_and_val, space=param_space, algo=tpe.suggest, 
                    max_evals=MAX_EVALS, trials=bayes_trials)

    n = 100
    q = 100

    radius = 1.0
    bias_var = 1.0
    Du_var = 1.0
    Bu_var = 1.0
    ESN = echo_state_network(n, 1, 1, q, 1.0, radius, phi, 200,
                             bias_var=bias_var, Du_var=Du_var,
                             Bu_var=Bu_var)


    # Initialize the reservoir
    ESN.ESN_init_IEE(diag_E=False, diag_P=False)
    # ESN.ESN_init_sr()
    # ESN.ESN_init_orthog(radius=0.9)

    ESN.train(train)
    train_perf = ESN.test(train)
    val_perf = ESN.test(val)
    test_perf = ESN.test(test)

    if test_perf > 1:
        test_perf = ESN.test(test)
        print("unstable system?")

    print("training performance: ", train_perf)
    print("val performance: ", val_perf)
    print("test performance: ", test_perf)

    eigs = np.linalg.eigvals(ESN.Bw)
    plt.plot(np.real(eigs), np.imag(eigs), '.')

    eigs = np.linalg.eigvals(ESN.A)
    plt.plot(np.real(eigs), np.imag(eigs), 'x')

    eigs = np.linalg.eigvals(ESN.A + ESN.Bw)
    plt.plot(np.real(eigs), np.imag(eigs), 'o')

    # eigs = np.linalg.eigvals(ESN.Atild)
    # plt.plot(np.real(eigs), np.imag(eigs), 's')

    plt.legend(['B eigs', 'A eigs',  'A + B eigs', 'Atild'])

    theta = np.linspace(0, 2*np.pi, 50)
    plt.plot(np.sin(theta), np.cos(theta))

    plt.show()
    print('~fin~')