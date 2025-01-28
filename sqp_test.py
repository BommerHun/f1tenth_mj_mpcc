import numpy as np
import os
import xml.etree.ElementTree as ET
import mujoco as mj
#from aiml_virtual.simulated_object.dynamic_object.controlled_object.car import Car
from car_model import Car
from control_model import Car_Control_Model
from aiml_virtual import scene, simulator, xml_directory
from trajectory_util import Trajectory_Marker, Spline_2D
from aiml_virtual.trajectory.car_trajectory import CarTrajectory
from F1TENTH_MPCC.f1tenth_mpcc import F1TENTHMJMPC_SQP
import math
import yaml
import copy
from trajectory_generators import eight
from marker import Marker

def quaternion_from_z_rotation(rotation_z):

    w = math.cos(rotation_z / 2)
    x = 0
    y = 0
    z = math.sin(rotation_z / 2)
    
    return f"{w} {x} {y} {z}"

phi0 = 3.14
car_pos = np.array([0, 0, 0.04999])
car_quat = quaternion_from_z_rotation(phi0)

path_points = np.array([
    [0,0],
    [1,1],
    [2,2],
    [3,3],
    [4,4],
    [5,5],]
    )

path_points, vel = eight()


def create_control_model(c_pos, c_quat):    
    scn = scene.Scene(os.path.join(xml_directory, "empty_checkerboard.xml"), save_filename=os.path.join("xml_models", "control_scene.xml"))

    c = Car_Control_Model(has_trailer= False)
    scn.add_object(c, pos="0 0 0", quat="1 0 0 0")
    sim = simulator.Simulator(scn)
    return sim.model, sim.data, scn.xml_name

def load_mpcc_params(filename = "mjpc_config.yaml"):
    with open(filename, 'r') as file:
        params = yaml.full_load(file)
        return params
    
    
if __name__ == "__main__":
    scn = scene.Scene(os.path.join(xml_directory, "empty_checkerboard.xml"), save_filename=os.path.join("xml_models", "main_scene.xml"))

    traj = CarTrajectory()
    traj.build_from_points_const_speed(path_points, path_smoothing=0.01, path_degree=4, const_speed=1.5)


    c = Car(has_trailer=False)

    scn.add_object(c, pos=f"{car_pos[0]} {car_pos[1]} {car_pos[2]}", quat=car_quat)
    m = Trajectory_Marker(x = path_points[:, 0], y = path_points[:,1])
    params = load_mpcc_params()
    horizon_marker = Marker(params["N"])
    scn.add_object(m)
    #c.controller = control  
    c.trajectory = traj
    scn.add_object(horizon_marker)
    sim = simulator.Simulator(scn)

    control_model, control_data , xml_name = create_control_model(car_pos, car_quat)

    controller = F1TENTHMJMPC_SQP(control_model, control_data, trajectory=traj, params=params, xml_path= xml_name)
    sim.model.opt.timestep = 0.01
    c.CTRL_FREQ = 20
    c.controller = controller
   
    qpos0 = np.zeros(c.model.nq)
    
    qpos0[:3] = car_pos
    qpos0[3] = 3.14/4
    with sim.launch():
        mj.mju_copy(c.data.qpos, qpos0)
        while sim.viewer.is_running():
            sim.tick()

            #c.data.qpos = c.controller.ocp_solver.get(1,'x')[:c.model.nq]
            for i in range(params["N"]):
                state = controller.ocp_solver.get(i, 'x')
                id = sim.model.body(f"mpcc_{i}").id

                id = sim.model.body_mocapid[id]
                sim.data.mocap_pos[id] = np.concatenate((state[:2], np.array([state[2]])))

            

