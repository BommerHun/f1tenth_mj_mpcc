import mujoco as mj
import numpy as np

from acados_template import AcadosOcp, AcadosOcpSolver

from aiml_virtual.controller.controller import Controller
from aiml_virtual.trajectory.car_trajectory import CarTrajectory
from aiml_virtual.simulated_object.dynamic_object.controlled_object.car import Car

from trajectory_util import Spline_2D

import casadi as cs
from scipy.interpolate import splev

import os

from types import SimpleNamespace

from mjpcipopt import F1TENTHMJMPC_IPOPT
import copy
class F1TENTHMJMPC_SQP(Controller):
    def __init__(self,
                model: mj.MjModel,
                data: mj.MjData,
                trajectory:CarTrajectory, 
                xml_path:str,
                params :dict):
        
        self.first_call = True
        self.model = model
        self.data = data

        self.nx = self.model.nv*2
        self.nu = self.model.nu

        #TODO The params file should contain:
        # - optimization weights
        # - state and input bounds
        # - Number of substeps that are required by mj
        self.params = params 
        
        self.dt = self.params["dt"]
        self.substep = self.params["substep"]
        self.model.opt.timestep = self.dt /self.substep
        self.N = self.params["N"]

        self._set_trajectory(trajectory)
        #####################################Extracting the optimization weights###########################################################

    
        self.weights = SimpleNamespace()
        self.weights.e_c = params["e_c"]
        self.weights.e_l = params["e_l"]
        self.weights.e_t = params["e_t"]
        self.weights.e_ss = params["e_ss"] # e_ss stands for steering-smoothness
        self.weights.e_sd = params["e_sd"] # e_sd motor-reference-smoothness


        #####################################Extracting vehicle parameters#################################################################

        self.vehicle_params = SimpleNamespace()
        self.vehicle_params.C_m1 = params["C_m1"]
        self.vehicle_params.C_m2 = params["C_m2"]
        self.vehicle_params.C_m3 = params["C_m3"]

        self.vehicle_params.tw = params["tw"]
        self.vehicle_params.wb = params["wb"]
        self.vehicle_params.wb = 2*Car.WHEEL_Y
        self.vehicle_params.tw = 2*Car.WHEEL_X
        ##########################################Extracting bounds########################################################################

        self.bounds = SimpleNamespace()

        self.bounds.delta_max = params["delta_max"]
        self.bounds.dot_delta_max = params["dot_delta_max"]
        self.bounds.d_max = params["d_max"]
        self.bounds.d_min = params["d_min"]
        self.bounds.theta_max = trajectory.length
        self.bounds.theta_min = 0
        self.bounds.dot_theta_max = params["dot_theta_max"]
        self.bounds.dot_theta_min = params["dot_theta_min"]
        
        #####################Write the xml-path, simulation timestep, and the number of substeps into the c source file####################
        self.xml_path = xml_path
        self.model_c_file = "mujoco_acados_dyn_param.c"
        self.write_parameters_to_c_source()
        #####################################Create casadi functions for the drivetrain and steering constraints###########################


        # Define variables
        d_l = cs.MX.sym('dl')
        d_r = cs.MX.sym('dr')

        ack_inv_left = cs.atan((cs.tan(d_l) * self.vehicle_params.wb) / (-0.5 * self.vehicle_params.tw * cs.tan(d_l) + self.vehicle_params.wb))
        ack_inv_right =  cs.atan((cs.tan(d_r) * self.vehicle_params.wb) / (0.5 * self.vehicle_params.tw * cs.tan(d_r) + self.vehicle_params.wb))



        self.ack_inv_left = cs.Function('inv_left', [d_l],[ack_inv_left])
        self.ack_inv_right =  cs.Function('inv_right', [d_r], [ack_inv_right])

        x = cs.MX.sym('in', 3)
        F_xi =x[0]
        v_x = x[1]
        v_y = x[2]
        
        self._motor_reference =  F_xi / self.vehicle_params.C_m1 + self.vehicle_params.C_m2 / self.vehicle_params.C_m1 *(cs.sqrt(v_x**2 + v_y**2))+self.vehicle_params.C_m3/self.vehicle_params.C_m1
        self._motor_reference = cs.Function('motor_reference', [x], [self._motor_reference])

        #####################################Expressing the objective function#############################################################

        x = cs.MX.sym('x', self.nx,1)
        u = cs.MX.sym('u', self.nu,1)

        # acados ocp model
        ocp = AcadosOcp()
        ocp.model.name = 'f1tenth_MJPC'

        # symbolics
        ocp.model.x = x
        ocp.model.u = u

        ocp.cost.cost_type = 'EXTERNAL'
        ocp.model.cost_expr_ext_cost = self._get_cost_expr(x,u)

        # Generic dynamics
        ocp.model.dyn_ext_fun_type = 'generic'
        ocp.model.dyn_generic_source = self.model_c_file
        ocp.model.dyn_disc_fun = 'disc_dyn_fun'
        ocp.model.dyn_disc_fun_jac = 'disc_dyn_fun_jac'

        #####################################Expressing the constraints####################################################################
        delta_lmax, delta_rmax = self._get_steering_angle(self.bounds.delta_max)
        delta_lmin, delta_rmin = self._get_steering_angle(-self.bounds.delta_max)
        ubx = np.array([ delta_rmax,delta_lmax,self.bounds.theta_max]) #Condition for the virtual state (theta)
        lbx = np.array([ delta_rmin,delta_lmin, 0.001])


    

        ubu = np.array((self.bounds.dot_delta_max, self.bounds.dot_delta_max,self.bounds.dot_theta_max)) #Condition for the change of the virtual state(theta)
        lbu = np.array((-self.bounds.dot_delta_max, -self.bounds.dot_delta_max,self.bounds.dot_theta_min)) 
        ocp.constraints.lbu = lbu
        ocp.constraints.ubu = ubu
        ocp.constraints.idxbu = np.array((0,3,6))

        ocp.constraints.lbx = lbx
        ocp.constraints.ubx = ubx
        ocp.constraints.idxbx = np.array((6, 9,12))
        ocp.constraints.x0 = np.zeros(self.nx)

        #ocp.constraints.lbx_0 = np.zeros(self.nx)
        #ocp.constraints.ubx_0 = np.zeros(self.nx)
        #ocp.constraints.idxbx_0 = np.arange(self.nx)


        ocp.model.con_h_expr,nc = self._get_h_expr(x,u)

        #Additional constraints
        ocp.constraints.lh = np.zeros(nc)
        ocp.constraints.uh = np.zeros(nc)
        ocp.dims.nh = nc

        ocp.constraints.lh[0] = self.bounds.d_min

        ocp.constraints.uh[0] = self.bounds.d_max


        # acados ocp opts
        ocp.solver_options.tf = self.N * self.dt
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.levenberg_marquardt = 0.1
        ocp.solver_options.nlp_solver_max_iter = 3000
        ocp.solver_options.tol = 1e-5
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.print_level = params["print_level"]

        # ocp.solver_options.levenberg_marquardt = 0.01
        
        ocp.solver_options.nlp_solver_tol_stat = 1e-5
        # ocp.solver_options.nlp_solver_tol_eq = 1e-3
        # ocp.solver_options.nlp_solver_tol_ineq = 1e-3

        ocp.solver_options.qp_solver_warm_start = 0
        # ocp.qp_solver_tol = 1e-3
                
        acados_dir = os.environ.get('ACADOS_SOURCE_DIR')
        ocp.solver_options.model_external_shared_lib_dir = os.path.join(acados_dir, "include", "mujoco-3.2.4", "lib")
        ocp.solver_options.model_external_shared_lib_name = 'mujoco'

        self.pre_controller = F1TENTHMJMPC_IPOPT(model, data, trajectory, params)
        self.ocp_solver = AcadosOcpSolver(ocp,json_file = f"c_generated_code/acados_ocp.json", verbose = False)


    def _get_h_expr(self, x, u):

        h = [] #List of constraints

        #Extracting steering variables
        d_l = x[9] #delta_left
        d_r = x[6] #delta_right

        dot_d_r = x[self.model.nv + 0]

        #Extracting drivetrain variables
        F_xi1 = u[1] #F_xi
        F_xi2 = u[2]
        F_xi3 = u[4]
        F_xi4 = u[5]
        
        v_x = x[self.model.nv+0]
        v_y = x[self.model.nv+1]



        #Drivetrain MIN-MAX
        h.append(self._motor_reference(cs.hcat([F_xi1,v_x, v_y])))

        #Steering ackermann geometry
        d_i_r = self.ack_inv_right(d_r)
        d_i_l = self.ack_inv_left(d_l)

        h.append(d_i_l-d_i_r)


        #Drivetrain equality
        h.append(F_xi1-F_xi2)
        h.append(F_xi2-F_xi3)
        h.append(F_xi3-F_xi4)

        return cs.vertcat(*h), len(h)

        

    def _get_steering_angle(self, delta):
        num = self.vehicle_params.wb * np.tan(delta)
        delta_left = np.atan(num / (self.vehicle_params.wb + (0.5 * self.vehicle_params.tw * np.tan(delta))))
        delta_right = np.atan(num / (self.vehicle_params.wb - (0.5 * self.vehicle_params.tw * np.tan(delta))))

        return delta_left, delta_right

    def _get_cost_expr(self,x,u):
        """Private function for expressing the cost for the trajectory tracking mpcc

        Args:
            x : current state
            u : current input

        Returns:
            cost (Casadi expression): Cost expression
        """
        #Extract opt variables
        point = cs.vertcat(x[0], x[1])
        theta = x[12]
        theta_dot = u[6]
        F_xi = u[0]

        point_r, v =  self.trajectory.get_path_parameters(theta, 0.001) #point: vertical, v: horizontal
        n = cs.hcat((v[:, 1], -v[:, 0])) #Creating a perpendicular vector

        e_c = cs.dot(n.T,(point_r-point)) #Contouring error
        e_l = cs.dot(v.T,(point_r-point)) #Lateral error


        #Let's take the the velocity of the steering of the left wheel (d l_delta/d t)
        #dot_right_steering = cs.fabs(x[self.model.nv + 6])
        #dot_steering = self.ack_inv_right(dot_right_steering)
    
        delta_input = self.ack_inv_right(u[0])

        steering_change = cs.fabs(delta_input)

        #Formulate cost
        cost = self.weights.e_c * e_c**2 + self.weights.e_l * e_l**2 + self.weights.e_ss * steering_change**2- self.weights.e_t*theta_dot + self.weights.e_sd * cs.fabs(F_xi**2)
        
        return cost
        
    def _set_trajectory(self, trajectory: CarTrajectory):
        #Transform ref trajectory
        t_end = trajectory.evol_tck[0][-1]
        s_end = splev(t_end, trajectory.evol_tck)
        t_eval=np.linspace(0, t_end, int(s_end+1))
        s=splev(t_eval, trajectory.evol_tck)
        (x,y) = splev(s, trajectory.pos_tck)

        points_list = []

        for i in range(len(x)):
            points_list.append([i, x[i], y[i]])

        self.trajectory = Spline_2D(np.array([[0,0,0],[1,1,1],[2,2,2]]))

        self.trajectory.spl_sx = cs.interpolant("traj", "bspline", [s], x)
        self.trajectory.spl_sy = cs.interpolant("traj", "bspline", [s], y)
        self.trajectory.L = s[-1]

    def compute_control(self, state, setpoint = None, time = None, **kwargs):
        qpos = copy.deepcopy(state["qpos"])
        qvel = copy.deepcopy(state["qvel"])
        qpos[7] = 0
        qpos[8] = 0
        qpos[10] = 0
        qpos[11] = 0
        # MPC initial state constraint
        x_0 = np.hstack((qpos,qvel))
        self.ocp_solver.set(0, 'lbx', x_0)
        self.ocp_solver.set(0, 'ubx', x_0)
        #self.ocp_solver.set(0, 'x', x_0)
        if self.first_call:
            delta,delta_vel, d, theta_vel = self.pre_controller.compute_control(state, None, None)
            #self.first_call = False
            print(f"precomputed input: {delta}, {d}")
            for i in range(self.N-1):
                self.ocp_solver.set(i+1, 'x', self.pre_controller.c_sol[(i+1)*self.nx : (i+2)*self.nx])
            self.ocp_solver.set(0, 'u', self.pre_controller.c_sol[self.N*self.nx : self.N*self.nx+self.nu])
            for i in range(self.N-2):
                self.ocp_solver.set(i+1, 'u', self.pre_controller.c_sol[self.N*self.nx + (i+1)*self.nu : self.N*self.nx+(i+2)*self.nu])
            for i in range(100):
                 status = self.ocp_solver.solve()

        for i in range(10):
            status = self.ocp_solver.solve()
        if status == 0:
            self.first_call = False
            ctrl = self.ocp_solver.get(1, 'u')
            pred_state = self.ocp_solver.get(1, 'x')
            d = self._motor_reference(cs.hcat((ctrl[1], pred_state[self.model.nq + 0],pred_state[self.model.nq + 1])))
            delta = self.ack_inv_left(pred_state[9])
            delta_vel = self.ack_inv_right(ctrl[0])
            theta_vel = ctrl[6]
            print(f"ctrl: {ctrl}, delta: {delta}, d: {d},d_theta: {theta_vel}, theta: {pred_state[12]}" )
        else:
            self.first_call = True
            ctrl = self.ocp_solver.get(1, 'u')
            pred_state = self.ocp_solver.get(1, 'x')
            d = self._motor_reference(cs.hcat((ctrl[1], pred_state[self.model.nq + 0],pred_state[self.model.nq + 1])))
            delta = self.ack_inv_left(pred_state[9])
            theta_vel = ctrl[6]
            delta_vel = self.ack_inv_right(ctrl[0])

        ctrl = self.ocp_solver.get(1, 'u')
        for i in range(self.N-1):
            self.ocp_solver.set(i, 'x', self.ocp_solver.get(i+1, 'x'))
        for i in range(self.N-2):
            self.ocp_solver.set(i, 'u', self.ocp_solver.get(i+1, 'u'))
        

        #d = self._motor_reference(cs.hcat((ctrl[1], pred_state[self.model.nq + 0],pred_state[self.model.nq + 1])))
        #delta = self.ack_inv_left(ctrl[0])
        #theta_vel = ctrl[6]
        #print(delta,d, theta_vel)
        return delta,delta_vel, d, theta_vel 

    def write_parameters_to_c_source(self):
        full_path = os.path.join(os.path.dirname(__file__),"..", self.model_c_file)
        start_row = 17
        # Read the file lines into a list
        try:
            with open(full_path, 'r') as file:
                lines = file.readlines()
        except FileNotFoundError as e:
            raise FileNotFoundError("\n\nC source file not found") from None

        # Modify the specified rows (start_row is 1-based)
        idx = start_row - 1
        lines[idx] = f"static char xml_path[200] = \"{self.xml_path}\";\n"
        idx += 1
        lines[idx] = f"static float timestep = {self.dt / self.substep};\n"
        idx += 1
        lines[idx] = f"static int substep = {self.substep};\n"

        # Write the modified content back to the file
        with open(full_path, 'w') as file:
            file.writelines(lines)