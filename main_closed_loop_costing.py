
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from pendulum_model import export_pendulum_ode_model, export_pendulum_ode_model_squashing
from utils import plot_pendulum, compute_terminal_control_law, plot_value_function_closed_loop
import numpy as np
import scipy.linalg
import casadi as ca

import matplotlib.pyplot as plt


def setup(x0, Fmax, N_horizon, Ttight, Tf):


    model = export_pendulum_ode_model_squashing(Fmax)
    model_ = export_pendulum_ode_model()

    Q_mat = np.diag([1e2, 1e2, 1e-2, 1e-2])
    R_mat = np.diag([1e-2])*Fmax**2
    P_mat, K_lqr = compute_terminal_control_law(model_, Q_mat, R_mat)

    cost_W = scipy.linalg.block_diag(Q_mat, R_mat)

    ocp = AcadosOcp()
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.dims.N = N_horizon

    ocp.cost.cost_type = 'CONVEX_OVER_NONLINEAR'
    ocp.cost.cost_type_e = 'CONVEX_OVER_NONLINEAR'

    squashed_u = ca.tanh(model.u)
    ocp.model.cost_y_expr = ca.vertcat(model.x, squashed_u)
    ocp.model.cost_y_expr_e = model.x

    r = ca.SX.sym('r', ny)
    r_e = ca.SX.sym('r_e', ny_e)
    ocp.model.cost_r_in_psi_expr = r
    ocp.model.cost_r_in_psi_expr_e = r_e

    scale = 1 - 1e-12
    ocp.model.cost_psi_expr = 0.5 * (r.T @ cost_W @ r) + model.p*(-ca.log((1-scale*r[nx:]))-ca.log((1+scale*r[nx:])))
    ocp.model.cost_psi_expr_e = 0.5 * (r_e.T @ P_mat @ r_e)

    ocp.cost.yref  = np.zeros((ny, ))
    ocp.cost.yref_e = np.zeros((ny_e, ))

    # set constraints
    # ocp.constraints.lbu = np.array([-Fmax])
    # ocp.constraints.ubu = np.array([+Fmax])
    # ocp.constraints.idxbu = np.array([0])

    ocp.constraints.C = K_lqr
    ocp.constraints.D = -np.eye(nu)
    ocp.constraints.lg = -10*np.array([Fmax])
    ocp.constraints.ug = 10*np.array([Fmax])

    ocp.constraints.x0 = x0
    ocp.parameter_values = np.array([100])
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.sim_method_newton_iter = 10
    ocp.solver_options.sim_method_num_stages = 3
    ocp.solver_options.sim_method_num_steps = 3
    ocp.solver_options.nlp_solver_max_iter = 100
    ocp.solver_options.nlp_solver_tol_stat = 1e-5
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.levenberg_marquardt = 1.

    ocp.solver_options.nlp_solver_type = 'SQP'

    ocp.solver_options.qp_solver_cond_N = N_horizon

    # set prediction horizon
    ocp.solver_options.tf = Tf

    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)

    sim = AcadosSim()

    sim.model = model_
    sim.solver_options.T = Tf/N_horizon
    sim.solver_options.integrator_type = 'IRK'
    sim.solver_options.num_stages = 3
    sim.solver_options.num_steps = 3
    sim.solver_options.newton_iter = 10
    acados_integrator = AcadosSimSolver(sim, json_file = 'acados_sim_' + model.name + '.json')

    dt = Tf/N_horizon
    N_tight = int(Ttight/dt)

    # set barrier parameters
    for n, p in enumerate(np.logspace(0, 3, N_horizon)):
        acados_ocp_solver.set(n, 'p', p)

    for n in range(N_tight, N_horizon):
        acados_ocp_solver.constraints_set(n, 'lg', -1e-5)
        acados_ocp_solver.constraints_set(n, 'ug', 1e-5)

    return acados_ocp_solver, acados_integrator, P_mat


def main(Fmax):

    x0 = np.array([0.0, np.pi, 0.0, 0.0])
    Nsim = 250

    Tf = .5
    Ttight = .3
    N_horizon = 100

    dt = Tf / N_horizon

    ax_value = None
    ax_traj = None

    for idx, Nupdate in enumerate([10, 20, 40]):

        ocp_solver, integrator, _ = setup(x0, Fmax, N_horizon, Ttight, Tf)

        nx = ocp_solver.acados_ocp.dims.nx
        nu = ocp_solver.acados_ocp.dims.nu

        simX = np.ndarray((Nsim+1, nx))
        simU = np.ndarray((Nsim+Nupdate, nu))
        simW = np.ndarray((Nsim+Nupdate, nu))

        simV = np.ndarray((Nsim,))

        simX[0,:] = x0

        for n, tau in enumerate(np.linspace(0, 1, N_horizon)):
            ocp_solver.set(n, 'x', (1-tau)*x0)

        # get initial guess
        for n in range(10):
            _ = ocp_solver.solve_for_x0(x0_bar = x0, fail_on_nonzero_status=False)

        # closed loop
        for i in range(Nsim):

            # solve ocp and get next control input
            _ = ocp_solver.solve_for_x0(x0_bar = simX[i, :], fail_on_nonzero_status=False)
            simV[i] = ocp_solver.get_cost()

            if i % Nupdate == 0:
                for n in range(Nupdate):
                    simW[i+n,:] = ocp_solver.get(n, 'u')
                    simU[i+n,:] = Fmax*ca.tanh(simW[i+n,:])

            # simulate system
            simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i,:])

        simU = simU[:Nsim]

        # plot results
        ts = np.linspace(0, (Tf/N_horizon)*Nsim, Nsim+1)
        label = f'$\\delta={Nupdate*dt}$'
        ax_traj = plot_pendulum(ts, Fmax, simU, simX, Nupdate, idx, plt_show=False, label=label, ax=ax_traj)

        ax_value = plot_value_function_closed_loop(ts[:-1], simV, label, ax_value)


    ax_traj[0].get_figure().tight_layout()
    ax_value.get_figure().tight_layout()
    plt.show()


if __name__ == '__main__':

    Fmax = 75

    main(Fmax)
