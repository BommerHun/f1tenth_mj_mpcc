from aiml_virtual.controller.controller import Controller
import mujoco as mj
import numpy as np
import casadi as cs
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import os
from scipy.interpolate import splev
from trajectory_util import Spline_2D
from aiml_virtual.trajectory.car_trajectory import CarTrajectory
class f1tenth_mjpc(Controller):
    """MPCC for F1TENTH"""

    def __init__(self,
                 model,
                 data,
                 horizon_lenght,
                 xml_path,
                 trajectory:CarTrajectory
                 ):
        self.model : mj.MjModel = model
        self.data: mj.MjData = data
        self.N = horizon_lenght
        
        self.nx = 2*np.shape(self.data.qvel)[0]
        self.nu = np.shape(self.data.ctrl)[0]

        self.nq = np.shape(self.data.qpos)[0]
        self.nv = np.shape(self.data.qvel)[0]

        # Time step for simulation and prediction
        self.model.opt.timestep = 0.05
        self.dt = 0.05

        # Quaternion states in the model are relative, this is the setpoint
        self.qpos0 = np.zeros(self.model.nq)
        self.qpos0[3] = 1  # quaternion real part

        # Last MPC solution
        self.x_n = None
        self.u_n = None

        self.set_trajectory(trajectory.evol_tck, trajectory.pos_tck)
        
        # Write xml path, time step, and reference parameters to the c source code
        self.xml_path = xml_path
        self.model_c_file = "mujoco_acados_dyn_param.c"
        self.write_parameters_to_c_source()

        
        # symbolic variables
        x = cs.MX.sym('x', self.nx, 1) # states
        u = cs.MX.sym('u', self.nu, 1) # controls
        delta_left = u[0]
        delta_right = u[3]

        #calculate steering using ackerman:

        wb = 0.24477
        tw= 0.24477        
        delta_in_left = cs.atan((-cs.tan(delta_left) * wb) / (0.5 * tw * cs.tan(delta_left) - wb))
        delta_in_right = cs.atan((-cs.tan(delta_right) * wb) / (0.5 * tw * cs.tan(delta_right) + wb))
        delta_steer = delta_in_left+delta_in_right
        fr = u[1]
        rl = u[2]
        fl = u[4]
        rr = u[5]

        delta_drivetrain = cs.mmax(cs.hcat([fr, rl, fl, rr]))-cs.mmin(cs.hcat([fr, rl, fl, rr]))
        # acados ocp model
        ocp = AcadosOcp()
        ocp.model.name = 'f1tenth'
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.dt *self.N
        # symbolics
        ocp.model.x = x
        ocp.model.u = u
        ocp.model.con_h_expr =     cs.vertcat(delta_steer,
                                                delta_drivetrain)


        ocp.cost.cost_type = "EXTERNAL"

        point = cs.vcat([x[0], x[1]])
        thetahat = x[12]
        thetahatdot =  u[6]

        ocp.model.cost_expr_ext_cost = self._cost_expr(point, theta=thetahat,thetahatdot=thetahatdot)

        #CONSTRAINTS:
        ocp.constraints.lh = np.array([0 ,0 ])
        ocp.constraints.uh = np.array([10^-3, 10^-3])

        ocp.constraints.lbx_0 = np.zeros(self.nx)
        ocp.constraints.ubx_0 = np.zeros(self.nx)
        ocp.constraints.idxbx_0 = np.arange(self.nx)


        # Generic dynamics
        ocp.model.dyn_ext_fun_type = 'generic'
        ocp.model.dyn_generic_source = self.model_c_file
        ocp.model.dyn_disc_fun = 'disc_dyn_fun'
        ocp.model.dyn_disc_fun_jac = 'disc_dyn_fun_jac'


        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'#'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI
        ocp.solver_options.nlp_solver_max_iter = 100
        ocp.solver_options.nlp_solver_tol_stat = 1e-3
        ocp.solver_options.levenberg_marquardt = 10.0
        ocp.solver_options.print_level = 0
        ocp.solver_options.qp_solver_iter_max = 1000
        ocp.code_export_directory = f"acados_c_generated/c_generated_code"
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'


        acados_dir = os.environ.get('ACADOS_SOURCE_DIR')
        ocp.solver_options.model_external_shared_lib_dir = os.path.join(acados_dir, "include", "mujoco", "lib")
        ocp.solver_options.model_external_shared_lib_name = 'mujoco'

        self.ocp_solver = AcadosOcpSolver(ocp)



    def _cost_expr(self, point,theta,thetahatdot):
        """
        Definition of the cost expression
        :param point: array containing x and y coordinates
        :param theta: path parameter
        :return: cost value (scalar)
        """

        e_c = self._cost_e_c(point,theta)
        e_l = self._cost_e_l(point,theta)


        q_con =50
        q_lat = 50
        q_theta = 1
    
        
        cost = e_c**2* q_con+e_l**2* q_lat-thetahatdot* q_theta
        
        
        return cost
    
    def _cost_e_c(self, point ,theta):
        """
        Contouring error function
        :param point: array containing x and y coordinates
        :param theta: path parameter(s)
        :return: contouring error
        """

        point_r, v =  self.trajectory.get_path_parameters(theta, 0) #point: vertical, v: horizontal
        n = cs.hcat((v[:, 1], -v[:, 0])) #Creating a perpendicular vector
        e_c = cs.dot(n.T,(point_r-point))
        return e_c


    def _cost_e_l(self, point, theta):
        """
        Lag error function
        :param point: array containing x and y coordinates
        :param theta: path parameter(s)
        :return: lag error
        """
        point_r, v = self.trajectory.get_path_parameters(theta, 0)
        e_l = cs.dot(v.T,(point_r-point))
        return e_l
    

    def set_trajectory(self, evol_tck, pos_tck):
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
        self.trajectory.spl_sx = cs.interpolant("traj", "bspline", [s], x)
        self.trajectory.spl_sy = cs.interpolant("traj", "bspline", [s], y)
        self.trajectory.L = s[-1]

    def compute_control(self, x0, setpoint,time, **kwargs):
        cur_qpos = x0[:14]

        qpos_rel = np.zeros(self.model.nv)
        qpos_rel[3] = 1
        mj.mj_differentiatePos(self.model, qpos_rel, 1, self.qpos0, cur_qpos)
        x0 = np.hstack([qpos_rel, x0[self.nv+1:]])
        x0[8] = 0
        x0[9] = 0
        x0[11] = 0
        x0[12] = 0
        
        print(x0)
        self.ocp_solver.set(0, 'lbx', x0)
        self.ocp_solver.set(0, 'ubx', x0)

        for i in range(self.N):
            self.ocp_solver.set(i, 'x', x0)    


        for i in range(self.N - 1):
            self.ocp_solver.set(i, 'u', np.zeros(7))
        self.ocp_solver.solve()
        ctrl = np.zeros(7)
        ctrl[1] = 0.05
        #ctrl[2] = 0.05
        #ctrl[4] = 0.05
        #ctrl[5] = 0.05
        #ctrl[6] = 0.05
        return ctrl    

    def write_parameters_to_c_source(self):
        full_path = os.path.join(os.path.dirname(__file__), self.model_c_file)
        start_row = 17
        # Read the file lines into a list
        try:
            with open(full_path, 'r') as file:
                lines = file.readlines()
        except FileNotFoundError as e:
            raise FileNotFoundError("\n\nC source file should be in the same folder as this file") from None

        # Modify the specified rows (start_row is 1-based)
        idx = start_row - 1
        lines[idx] = f"static char xml_path[200] = \"{self.xml_path}\";\n"
        idx += 1
        lines[idx] = f"static double timestep = {self.dt};\n"

        # Write the modified content back to the file
        with open(full_path, 'w') as file:
            file.writelines(lines)