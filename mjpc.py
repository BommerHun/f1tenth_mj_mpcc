from aiml_virtual.controller.controller import Controller
import mujoco as mj
import numpy as np
import casadi as cs
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import os

class f1tenth_mjpc(Controller):
    """MPCC for F1TENTH"""

    def __init__(self,
                 model,
                 data,
                 horizon_lenght,
                 xml_path,
                 trajectory
                 ):
        self.model : mj.MjModel = model
        self.data: mj.MjData = data
        self.N = horizon_lenght
        
        self.nx = np.shape(self.data.qvel)[0]
        self.nu = np.shape(self.data.ctrl)[0]

        self.nq = np.shape(self.data.qpos)
        self.nv = np.shape(self.data.qvel)

        # Time step for simulation and prediction
        self.model.opt.timestep = 0.05
        self.dt = 0.05

        # Quaternion states in the model are relative, this is the setpoint
        self.qpos0 = np.zeros(self.model.nq)
        self.qpos0[3] = 1  # quaternion real part

        # Last MPC solution
        self.x_n = None
        self.u_n = None


        # Write xml path, time step, and reference parameters to the c source code
        self.xml_path = xml_path
        self.model_c_file = "mujoco_acados_dyn_param.c"
        self.write_parameters_to_c_source()


        # symbolic variables
        x = cs.SX.sym('x', self.nx, 1) # states
        u = cs.SX.sym('u', self.nu, 1) # controls

        # acados ocp model
        ocp = AcadosOcp()
        ocp.model.name = 'drone_payload'
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.dt
        # symbolics
        ocp.model.x = x
        ocp.model.u = u

        ocp.cost.cost_type = "EXTERNAL"

        ocp.model.cost_expr_ext_cost = 0 #TODO


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

    def compute_control(self, *args, **kwargs):
        return super().compute_control(*args, **kwargs)
    

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