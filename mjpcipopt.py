import numpy as np
import mujoco as mj
import cyipopt as ci
from aiml_virtual.controller.controller import Controller as BaseController
import copy
import matplotlib.pyplot as plt
import sys
import re
import os
import threading
import time as time_ext
import sympy as sp
from aiml_virtual.simulated_object.dynamic_object.controlled_object.car import Car
from aiml_virtual.trajectory.car_trajectory import CarTrajectory
from trajectory_util import Spline_2D
import casadi as cs
from scipy.interpolate import splev
import yaml
from util.MPCC_plotter import MPCC_plotter

class OptProblem(ci.Problem):
    def __init__(self, model= None, data= None, trajectory:CarTrajectory= None, N= None, qpos_0= None, weights= None, solver_params = None, bounds = 0):
        #Contents:
        """
        -nx; nu; nc; nq through the model and data
        -N: prediction horizon
        -cl and cu: lower and upper constraints
        -trajectory: for evaluating the errors
        -opt_params: ?????
        qpos0: initial state
        -weights: opt weights
        -vehicle_params: vehicle parameters
        """
        
        #####################################Creating class variables######################################################################
        self.N = N
        self.model = model
        self.data = data
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nx = 2*self.nv
        self.nu = self.model.nu
        self.fd_centered = True  # 0: forward, 1: centered difference
        self.fd_eps = 10**-6
        self.qpos0 = qpos_0
        self.solver_params = solver_params
        self.bound = bounds
        self.wc ,self.wl,self.wt =  weights

        self._set_trajectory(trajectory.evol_tck, trajectory.pos_tck) #Convert the default CarTrajectry to the casadi implementation

        #TODO LOTS OF TODO
        x_lb, x_ub, u_lb, u_ub = self.bound
        lbx = self.N * x_lb.tolist() + (self.N - 1) * u_lb.tolist()
        ubx = self.N * x_ub.tolist() + (self.N - 1) * u_ub.tolist()


        #We have tons of constraints:
        """
            -Initial condition & dynamics constraint (N)
            -Drivetrain constraints-> equalitiy 3*(N-1)
            -Steering Constraints-> ackerman steering N-1
            - """
        self.nc = self.N*self.nx+(self.N-1)*(3)+(self.N-1)
        

        cl = self.nc * [0]
        cu = self.nc * [0]
        
        # setup optimization
        n_opt = self.N * self.nx + (self.N - 1) * self.nu
        n_con = self.nc
        self.jac_struct = self.init_jacobian_structure()
        
        super(OptProblem, self).__init__(n_opt, n_con, lb=lbx, ub=ubx, cl=cl, cu=cu)
        #####################################Variables for the inverse ackerman############################################################

        self.wb = 2*Car.WHEEL_Y
        self.tw = 2*Car.WHEEL_X

        #####################################Creating the inverse ackerman formulas:#######################################################
        # Define variables
        d_l = cs.MX.sym('dl')
        d_r = cs.MX.sym('dr')


        self.ack_inv_left = cs.atan((-cs.tan(d_l) * self.wb) / (0.5 * self.tw * cs.tan(d_l) - self.wb))
        self.ack_inv_right =  -cs.atan((-cs.tan(d_r) * self.wb) / (0.5 * self.tw * cs.tan(d_r) + self.wb))
        
        #####################################Expressing the derivates of the inverse ackermann equations:##################################
        self.dot_ack_inv_left = cs.gradient(self.ack_inv_left, d_l)
        self.dot_ack_inv_right = cs.gradient(self.ack_inv_right, d_r)

        #####################################Creating the callable cassadi functions:######################################################

        self._ack_inv_left = cs.Function('inv_left', [d_l],[self.ack_inv_left])
        self._ack_inv_right =  cs.Function('inv_right', [d_r], [self.ack_inv_right])
        self._dot_ack_inv_left = cs.Function('dot_inv_left', [d_l], [self.dot_ack_inv_left])
        self._dot_ack_inv_right = cs.Function('dot_inv_right', [d_r], [self.dot_ack_inv_right])


        #####################################Expressing the objective function#############################################################
        cost = 0
        x = cs.MX.sym('x', n_opt)
        for k in range(self.N):
            theta = x[k*self.nx + 13]
            point = cs.vertcat(x[k*self.nx + 0], x[k*self.nx + 1]) 

            point_r, v =  self.trajectory.get_path_parameters(theta = theta, theta_0= 0.05) 
            n = cs.hcat((v[:, 1], -v[:, 0])) #Creating a perpendicular vector

            e_c = cs.dot(n.T,(point_r-point))
            e_l = cs.dot(v.T,(point_r-point))

            cost += e_c*self.wc + e_l*self.wl

        k_0 = self.N*self.nx
        for k in range(self.N-1):
            thetadot = x[k_0 + k*self.nu + 6]
            cost-= thetadot*self.wt

        self._objective_ = cs.Function('objective', [x], [cost])
        grad = cs.gradient(cost, x)
        self._gradient_ = cs.Function('gradient', [x], [grad])


        #####################################Setting the solver problem settings##################################

        self.add_option("max_iter", solver_params["max_iter"])
        self.add_option("sb", "yes")
        self.add_option("print_level", solver_params["print_level"])
        self.add_option("print_timing_statistics", "no")
        self.add_option('tol', solver_params["tol"])
        self.add_option('mu_strategy', 'adaptive')

    def objective(self, x):
        """Returns the scalar value of the objective given x.

        Args:
            x
        """
        cost = self._objective_(x)
       
        return cost

    def gradient(self,x):
        """Returns the gradient of the objective given x.

        Args:
            x
        """
        gradient = self._gradient_(x)
        return gradient

    def solve(self, x_0, x_init):
        self.x_0 = x_0
        x, info = super(OptProblem, self).solve(x_init)
        return x, info
    def constraints(self, x):
        """Return constaints

        Args:
            x: optimisation variables
        """
        #TODO: create a symbolic function for evaulating the steering input of the bicycle model from the vehicle's current steering inputs \TICK\
        #list of constraints: 
        #TODO: Initial condition \TICK\
        #TODO: Dynamics constraint 1-N \TICK\
        #TODO: drivetrain: u_d1 - ud_2 =0, ud_2 - ud_3 = 0, ud_3 - ud_4 = 0 \TICK\
        #TODO: ack_inv(delta_left)-ack_inv(delta_right) = 0 \TICK\


        #delta_in_left = cs.atan((-cs.tan(delta_left) * wb) / (0.5 * tw * cs.tan(delta_left) - wb))
        #delta_in_right = cs.atan((-cs.tan(delta_right) * wb) / (0.5 * tw * cs.tan(delta_right) + wb))

        constraints = np.array([])

        k_0 = self.N * self.nx # shows the end of the state trajectory

        """Dynamics Constraints"""
        constraints = np.append(constraints, x[:self.nx] - self.x_0) #Initial condition
        for k in range(self.N-1):
            constraints = np.append(constraints, x[(k+1)*self.nx:(k+2)*self.nx]-self.dyn_step(x[k*self.nx: (k+1)*self.nx], x[k_0 + k*self.nu : k_0 + (k+1)*self.nu]))

        """Drivetrain Constraints: EQUALITY"""
        #INDEXING OF the actuators: Fr:1 Rl: 2 Fl: 4 Rr: 5-> these indexes are shifted by k_0, to the end of the STATE trajectory
        
        for k in range(self.N-1):
            constraints = np.append(constraints, x[k_0+ k*self.nu +1]- x[k_0+ k*self.nu +2])

        for k in range(self.N-1):
            constraints = np.append(constraints, x[k_0+ k*self.nu +2]- x[k_0+ k*self.nu +4])

        for k in range(self.N-1):
            constraints = np.append(constraints, x[k_0+ k*self.nu +4]- x[k_0+ k*self.nu +5])

        """Ackermann Steering Constraints"""
        #INDEXING OF the actuators: Left:0 Right:3-> these indexes are shifted by k_0, to the end of the STATE trajectory

        for k in range(self.N-1):
            #Evalutating the inverse ackermann steering equation in the current inputs
            d_l = x[k_0+ k*self.nu +0]
            d_r = x[k_0+ k*self.nu +3]

            d_i_l, d_i_r = self._inverse_ackerman_steering(d_l, d_r) # d_i_l: delta_input_left d_i_r: delta_input_right
            constraints = np.append(constraints, d_i_l-d_i_r) #The constraint is written: d_i_l-d_i_r = 0 !!!!!!!!
      

        #Return collected contraints
        """FINAL FORM OF THE CONSTRAINT VECTOR:"""
        return constraints


    def _inverse_ackerman_steering(self, d_l, d_r):
        """Basic function for calculating the current steering input of the bicycle model based on the current actuator values

        Args:
            d_l: left wheel actuator position
            d_r: right wheel actuator position
        
        Returns:
            d_l_in, 
            d_r_in
        """
        #delta_in_left = np.atan((-np.tan(d_l) * self.wb) / (0.5 * self.tw * np.tan(d_l) - self.wb))
        #delta_in_right = np.atan((-np.tan(d_r) * self.wb) / (0.5 * self.tw * np.tan(d_l) + self.wb))

        delta_in_left = self._ack_inv_left(d_l)
        delta_in_right = self._ack_inv_right(d_r)

        #delta_in_left = d_l
        #delta_in_right = -d_r
        return delta_in_left, delta_in_right
    

    def _der_inverse_ackermann_steering(self, d_l, d_r):
        delta_in_left = self._dot_ack_inv_left(d_l)
        delta_in_right = self._dot_ack_inv_right(d_r)

        #delta_in_left = 1
        #delta_in_right = 1
        return delta_in_left, delta_in_right

    def jacobian(self, x):
        """Return jacobian of the constraints

        Args:
            x: optimisation variables
        """
        con_jac = np.zeros((self.nc, self.N * self.nx + (self.N - 1) * self.nu))
        k_0 = self.N * self.nx  #stores the end of the state trajectory (beginning of the input trajectory)
        N_0 = 0         #stores the correct index of constraint

        """Initial state constraint:"""
        jac_0 = np.zeros((self.nx, self.N*self.nx + (self.N-1)*self.nu))

        jac_0[:, :self.nx] = np.eye(self.nx, self.nx)

        con_jac[:self.nx, :] = jac_0

        N_0 += self.nx
        
        
        jac = np.zeros((self.nx, self.N * self.nx + (self.N - 1) * self.nu))


        """Dynamics Constraint"""

        for k in range(self.N-1):          
            A, B = self.dyn_jac(x[k * self.nx : (k + 1) * self.nx], x[k_0 + k * self.nu : k_0 + (k + 1) * self.nu])
            jac = np.zeros((self.nx, self.N * self.nx + (self.N - 1) * self.nu))
            A, B = self.dyn_jac(x[k * self.nx : (k + 1) * self.nx], x[k_0 + k * self.nu : k_0 + (k + 1) * self.nu])

            jac[:, k * self.nx : (k + 1) * self.nx] = -A
            jac[:, k_0 + k * self.nu : k_0 + (k + 1) * self.nu] = -B
            jac[:, (k + 1) * self.nx : (k + 2) * self.nx] = np.eye(self.nx)
            con_jac[(k+1) * self.nx : (k + 2) * self.nx, :] = jac

        N_0 =N_0+  self.nx*(self.N-1)

        """Drivetrain Constraints: EQUALITY""" #row48-50
        for k in range(self.N-1):
            con_jac[N_0+k, k_0 + k*self.nu + 1] = 1
            con_jac[N_0+k, k_0 + k*self.nu + 2] = -1
        N_0 +=self.N-1

        for k in range(self.N-1):
            con_jac[N_0+k, k_0 + k*self.nu + 2] = 1
            con_jac[N_0+k, k_0 + k*self.nu + 4] = -1
        N_0+= self.N-1

        for k in range(self.N-1):
            con_jac[N_0+k, k_0 + k*self.nu + 4] = 1
            con_jac[N_0+k, k_0 + k*self.nu + 5] = -1
        N_0 +=self.N-1


        """Ackermann Steering Constraints"""#row52

        for k in range(self.N-1):
            d_l = x[k_0 + k*self.nu + 0]
            d_r = x[k_0 + k*self.nu + 3]

            d_inv_l, d_inv_r = self._der_inverse_ackermann_steering(d_l=d_l, d_r= d_r)
            con_jac[N_0+k, k_0 + k*self.nu + 0] = d_inv_l
            con_jac[N_0+k, k_0 + k*self.nu + 3] = -d_inv_r
        N_0 += self.N-1


        row, col = self.jacobianstructure()

        return con_jac[row, col]
        

    def dyn_step(self, x_cur, u_cur):
        qpos_err = x_cur[:self.model.nv]
        qpos_actual = self.qpos0.copy()
        mj.mj_integratePos(self.model, qpos_actual, qpos_err, 1)
        self.data.qpos = qpos_actual
        self.data.qvel = x_cur[self.model.nv:]
        self.data.ctrl = u_cur
        #for _ in range(int(self.dt / self.model.opt.timestep)):
        mj.mj_step(self.model, self.data)
        mj.mj_differentiatePos(self.model, qpos_err, 1, self.qpos0, self.data.qpos)
        return np.hstack((qpos_err, self.data.qvel))

    def dyn_jac(self, x_cur, u_cur):
        qpos_err = x_cur[:self.model.nv]
        qpos_actual = self.qpos0.copy()
        mj.mj_integratePos(self.model, qpos_actual, qpos_err, 1)
        self.data.qpos = qpos_actual
        self.data.qvel = x_cur[self.model.nv:]
        self.data.ctrl = u_cur
        A = np.zeros((2*self.model.nv, 2*self.model.nv))
        B = np.zeros((2*self.model.nv, self.model.nu))
        mj.mjd_transitionFD(self.model, self.data, self.fd_eps, self.fd_centered, A, B, None, None)
        return A, B
    

    def init_jacobian_structure(self):  
        """Create a sparse matrix to hold the jacobian structure"""  
        con_jac = np.zeros((self.nc, self.N*self.nx + (self.N-1)*self.nu))


        """Initial state constraint:"""
        jac_0 = np.zeros((self.nx, self.N*self.nx + (self.N-1)*self.nu))
        jac_0[:, :self.nx] = np.eye(self.nx, self.nx)


        con_jac[:self.nx, :] = jac_0


        """Dynamics Constraint"""
        k_0 = self.N * self.nx #stores the end of the state trajectory (beginning of the input trajectory)
        jac = np.zeros((self.nx, self.N * self.nx + (self.N - 1) * self.nu))
        N_0 = self.nx #stores the correct index of constraint
        for k in range(self.N-1):          
            jac[:, :] = 0 #zeroing the whole matrix
            jac[:, k * self.nx : (k + 1) * self.nx] = 1
            jac[:, k_0 + k * self.nu : k_0 + (k + 1) * self.nu] = 1
            jac[:, (k + 1) * self.nx : (k + 2) * self.nx] = np.eye(self.nx)
            con_jac[(k+1) * self.nx : (k + 2) * self.nx, :] = jac
        N_0 =N_0+  self.nx*(self.N-1)


        """Drivetrain Constraints: EQUALITY""" #row48-50
        for k in range(self.N-1):
            con_jac[N_0+k, k_0 + k*self.nu + 1] = 1
            con_jac[N_0+k, k_0 + k*self.nu + 2] = 1
        N_0 +=self.N-1

        for k in range(self.N-1):
            con_jac[N_0+k, k_0 + k*self.nu + 2] = 1
            con_jac[N_0+k, k_0 + k*self.nu + 4] = 1
        N_0+= self.N-1

        for k in range(self.N-1):
            con_jac[N_0+k, k_0 + k*self.nu + 4] = 1
            con_jac[N_0+k, k_0 + k*self.nu + 5] = 1
        N_0 +=self.N-1


        """Ackermann Steering Constraints"""#row52
        for k in range(self.N-1):
            con_jac[N_0+k, k_0 + k*self.nu + 0] = 1
            con_jac[N_0+k, k_0 + k*self.nu + 3] = 1
        N_0 += self.N-1



        """print(con_jac)
        for i in range(self.nc):
            if i < self.N*self.nx:
                continue
            i = i+1"""
        return np.nonzero(con_jac)
    
    def jacobianstructure(self):
        """Return the jacobian structure"""
        return self.jac_struct

    def _set_trajectory(self, evol_tck, pos_tck):
        #Transform ref trajectory
        t_end = evol_tck[0][-1]
        s_end = splev(t_end, evol_tck)
        t_eval=np.linspace(0, t_end, int(s_end+1))
        s=splev(t_eval, evol_tck)
        (x,y) = splev(s, pos_tck)

        points_list = []

        for i in range(len(x)):
            points_list.append([i, x[i], y[i]])

        self.trajectory = Spline_2D(np.array([[0,0,0],[1,1,1],[2,2,2]]))
        self.trajectory.spl_sx = cs.interpolant("trajx", "bspline", [s], x)
        self.trajectory.spl_sy = cs.interpolant("trajy", "bspline", [s], y)
        self.trajectory.L = s[-1]

    
class F1TENTHMJPC(BaseController):
    """Mujoco based MPCC controller for the F1TENTH platform"""

    def __init__(self,
                 model,
                 data,
                 trajectory: CarTrajectory,
                 params):
        """#TODO
        """
        self.model = model
        self.data : mj.MjData = data
        self.nx =2* self.model.nv
        self.nu = self.model.nu

        #Time step for the simulation
        self.model.opt.timestep = params["dt"]
        self.dt = params["dt"]

        # Expressing the boundaries
        d_max = params["d_max"]
        d_min = params["d_min"]
        theta_max = trajectory.length
        theta_dot_max = params["dtheta_max"]
        theta_dot_min = params["dtheta_min"]
        delta_max = params["delta_max"]


        ubx =  np.ones(self.nx)* 10**20
        ubx[13] = theta_max

        ubu = np.ones( self.nu)*10**20
        ubu[0] = delta_max
        ubu[3] = delta_max
        ubu[6] = theta_dot_max
        ubu[1] = d_max

        lbx =  np.ones(self.nx)* 10**-20
        lbx[13] = theta_max

        lbu = np.ones(self.nu)*10**-20
        lbu[0] = -delta_max
        lbu[3] = -delta_max
        lbu[6] = theta_dot_min
        lbu[1] = d_min
        
        bounds = (lbx, ubx, lbu, ubu)
        # Quaternion states are relative, this is the setpoint
        self.qpos0 = np.zeros(self.model.nq)
        self.qpos0[3] = 1

        self.N = params["N"]
        
        # MPC solver parameters
        solver = {
            "print_level": params["print_level"],
            "max_iter": params["max_iter"],
            "tol": params["tol"]
        }

        #Extracting optimisation weights:
        weights = (params["e_c"], params["e_l"], params["e_t"])


        #Create optimization params for opt_params #TODO
        prob_params = (self.model, self.data, trajectory, self.N, self.qpos0, weights, solver, bounds)

        self.problem = OptProblem(*prob_params)

        self.plotter = MPCC_plotter()
        s = np.linspace(0, trajectory.length, 1000)
        self.plotter.set_ref_traj(np.array(self.problem.trajectory.spl_sx(s)), np.array(self.problem.trajectory.spl_sy(s)))
        self.plotter.show()


    def compute_control(self, state, setpoint, time):
        # Extract states
        qpos_rel = np.zeros(self.model.nv)
        cur_qpos = copy.deepcopy(state["qpos"])
        mj.mj_differentiatePos(self.model, qpos_rel, 1, self.qpos0, cur_qpos)
        x_0 = np.hstack((qpos_rel, state["qvel"]))


        x, info = self.problem.solve(x_0, np.ones(self.N*self.nx + (self.N-1)*self.nu))
        ctrl = x[self.N*self.nx+self.nu : self.N*self.nx+self.nu*2]
        #ctrl[0] = ctrl[3]
        print(ctrl)
        new_x = x[0:self.N*self.nx:self.nx]
        new_y = x[0:self.N*self.nx:self.nx]
        self.plotter.update_plot(new_x, new_y)
        return ctrl