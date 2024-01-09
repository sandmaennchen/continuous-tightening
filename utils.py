
import matplotlib.pyplot as plt
import numpy as np
from acados_template import latexify_plot
import casadi as ca
import scipy
from matplotlib import cm


latexify_plot()

def plot_pendulum(t, u_max, U, X_true, Nupdate, i, plt_show=True, label=None, ax=None):
    """
    Params:
        shooting_nodes: time values of the discretization
        u_max: maximum absolute value of u
        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X_true: arrray with shape (N_sim, nx)
    """

    if ax is None:
        fix, ax = plt.subplots(ncols=1, nrows=3, sharex=True, figsize=(4.2, 6.5))


    color = f'C{i}'
    N_sim = X_true.shape[0]

    delta = t[1]

    N = int(N_sim/Nupdate)

    idx = 0
    for n in range(N):
        ax[0].plot(np.append(t[idx:idx+Nupdate], t[idx+Nupdate]-0.5*delta), np.append(U[idx:idx+Nupdate], U[idx+Nupdate-1]), alpha=0.7, color=color)
        idx += Nupdate

    ax[0].set_ylabel('$u$')
    ax[0].hlines(u_max, t[0], t[-1], linestyles='dashed', alpha=0.75, color='gray')
    ax[0].hlines(-u_max, t[0], t[-1], linestyles='dashed', alpha=0.75, color='gray')
    ax[0].set_ylim([-1.2*u_max, 1.2*u_max])
    ax[0].set_xlim(t[0], t[-1])
    ax[0].grid(True)

    states_lables = ['$p$', r'$\theta$', '$v$', r'$\dot{\theta}$']

    for i in range(2):
        ax[i+1].plot(t, X_true[:, i], label=label, alpha=0.7)
        ax[i+1].set_ylabel(states_lables[i])
        ax[i+1].grid(True)
        ax[i+1].set_xlim(t[0], t[-1])

    ax[-1].legend()
    ax[-1].set_xlabel('$t$')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.1)

    if plt_show:
        plt.show()

    return ax


def plot_value_function_closed_loop(ts, Vs, label, ax):

    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4.2, 2.8))

    ax.plot(ts, Vs, label=label)

    ax.set_xlim(ts[0], ts[-1])
    ax.set_xlabel('$t$')
    ax.set_ylabel(r'$V_\mathrm{x}(x^*_{\delta}(t, x_0))$')
    ax.legend()
    ax.set_ylim(bottom=0.12)
    ax.set_yscale('log')
    ax.grid(True)
    return ax


def compute_terminal_control_law(model, Q_mat, R_mat):

    x0 = np.zeros(model.x.shape)
    u0 = np.zeros(model.u.shape)
    A_mat_fun = ca.Function('A', [model.x, model.u], [ca.jacobian(model.f_expl_expr, model.x)])
    B_mat_fun = ca.Function('B', [model.x, model.u], [ca.jacobian(model.f_expl_expr, model.u)])
    B_mat = B_mat_fun(x0, u0)
    A_mat = A_mat_fun(x0, u0)

    P_mat  = scipy.linalg.solve_continuous_are(A_mat, B_mat, Q_mat, R_mat)

    K_lqr = 0.8*np.linalg.solve(R_mat, B_mat.T @ P_mat)
    # K_lqr = np.array([[-1.25      , 32.43543044, -2.3486307 ,  8.97093847]])
    K_mat = np.array([[-.5      , 30., -2 ,  9]])

    A_cl_mat = A_mat - B_mat @ K_mat

    eig_max_A = np.real(np.max(scipy.linalg.eigvals(A_cl_mat)))

    kappa = -0.5*eig_max_A

    P_mat_conservative  = scipy.linalg.solve_continuous_are(A_cl_mat + kappa*np.eye(A_cl_mat.shape[0]), 0*B_mat, Q_mat + K_mat.T @ R_mat @ K_mat, R_mat)

    eigvals = np.linalg.eigvals(P_mat_conservative)
    print('eigenvalues of P', eigvals)

    return P_mat_conservative, K_mat


def plot_control_cost(R, taus, Fmax):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.2, 2.5))

    vs = np.expand_dims(np.linspace(-2, 2, 200), axis=0)
    us = np.tanh(vs)

    colors = cm.get_cmap('Greys_r', len(taus)+3)

    # mark -0.9Fmax, 0.9Fmax interval
    ax.axvspan(-1.47222, 1.47222, alpha=0.15, color='olive')
    for n, tau in enumerate(taus):
        ls = tau*(-np.log((1-us))-ca.log((1+us))) + Fmax**2*R*us**2
        ax.plot(vs.ravel(), ls.full().ravel(), c=colors(n), alpha=0.9)

    ax.grid()
    ax.set_xlim(left=vs[0, 0], right=vs[0, -1])
    ax.set_ylim(bottom=-5, top=80)
    ax.set_xlabel('$w$')
    ax.set_ylabel(r'$l_{\mathrm{w}}(w, \tau)$')

    fig.tight_layout()
    plt.show()


def main_barrier():

    n_grid = 601
    x_lim = 5
    l_hess = 0.2
    colors = cm.get_cmap('Greys_r', 3)

    barrier_param = 2

    plt.figure(figsize=(4, 2.25))

    xs = np.linspace(-x_lim, x_lim, n_grid)

    l_vals =  l_hess*xs**2
    b_vals = -np.log(x_lim - xs) - np.log(x_lim+xs) + np.log(x_lim) + np.log(x_lim)
    b_vals[0] = 1e10
    b_vals[-1] = 1e10

    plt.plot(xs, l_vals, label=r'$\bar{l}(u)$', color=colors(0))
    plt.plot([xs[0], xs[0]], [l_vals[0], 1e10], color=colors(0), linestyle='dashed')
    plt.plot([xs[-1], xs[-1]], [l_vals[-1], 1e10], color=colors(0), linestyle='dashed')
    plt.plot(xs, l_vals + barrier_param*b_vals, label=r'$\bar{\tilde{l}}(u)$', color=colors(1))

    plt.grid()
    plt.legend(loc='upper left', framealpha=1.)
    plt.xlim([-6, 6])
    plt.ylim([-0.8, 17])
    plt.xlabel(r'$u$')
    plt.tight_layout()


def main_progressive_tightening():

    n_grid = 201
    x_lims = [5, 4, 3]
    l_hess = [0.2, 0.4, 0.8]
    colors = cm.get_cmap('Greys', len(x_lims)+1)

    plt.figure(figsize=(3.8, 2.25))

    for i, (x_lim, b) in enumerate(zip(x_lims, l_hess)):
        xs = np.linspace(-x_lim, x_lim, n_grid)

        vals =  b*xs**2

        plt.plot(xs, vals, label=r'$\bar l(x, \tau_{%d})$' % i, color=colors(i+1))
        plt.plot([xs[0], xs[0]], [vals[0], 1e10], color=colors(i+1), linestyle='dashed')
        plt.plot([xs[-1], xs[-1]], [vals[-1], 1e10], color=colors(i+1), linestyle='dashed')
        plt.fill_between(np.hstack((xs, [xs[-1], xs[0]])), np.hstack((vals,[1e10, 1e10])), color='k', alpha=0.09)

    plt.grid()
    plt.legend(loc='upper left', framealpha=1.)
    plt.xlim([-6, 6])
    plt.ylim([-0.8, 17])
    plt.xlabel(r'$x$')
    plt.tight_layout()


if __name__ == '__main__':

    # main_barrier()
    main_progressive_tightening()

    R_mat = np.diag([1e-2])
    Fmax = 75
    taus = np.logspace(0, 3, 8)
    plot_control_cost(R_mat, taus, Fmax=Fmax)

    plt.show()