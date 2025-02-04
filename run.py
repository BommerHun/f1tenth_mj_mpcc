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
from mjpcipopt import F1TENTHMJMPC_IPOPT
import math
import yaml
import copy
from trajectory_generators import eight

def quaternion_from_z_rotation(rotation_z):

    w = math.cos(rotation_z / 2)
    x = 0
    y = 0
    z = math.sin(rotation_z / 2)
    
    return f"{w} {x} {y} {z}"

phi0 = 3.14/2

car_pos = np.array([0, 0, 0.04999])
car_quat = quaternion_from_z_rotation(phi0)
path_points, vel = eight()


def create_control_model(c_pos, c_quat):
    scn = scene.Scene(os.path.join(xml_directory, "empty_checkerboard.xml"), save_filename=os.path.join("xml_models", "control_scene.xml"))

    c = Car_Control_Model()
    scn.add_object(c, pos=f"{car_pos[0]} {car_pos[1]} {car_pos[2]}", quat=car_quat)
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

    c_model, c_data, c_scene = create_control_model(c_pos= car_pos, c_quat=car_quat)

    c = Car(has_trailer=False)


    scn.add_object(c, pos=f"{car_pos[0]} {car_pos[1]} {car_pos[2]}", quat=car_quat)

    m = Trajectory_Marker(x = path_points[:, 0], y = path_points[:,1])
    params = load_mpcc_params()
    #c.controller = control  
    c.trajectory = traj
    scn.add_object(m)
    sim = simulator.Simulator(scn)

    control_model = copy.deepcopy(sim.model)
    control_data = copy.deepcopy(sim.data)

    controller = F1TENTHMJMPC_IPOPT(control_model, control_data, trajectory=traj, params=params)
    sim.model.opt.timestep = 0.01

    c.controller = controller

    qpos0 = np.zeros(c.model.nq)

    qpos0[:3] = car_pos
    qpos0[3] = 3.14/4
    with sim.launch():
        mj.mju_copy(c.data.qpos, qpos0)
        while sim.viewer.is_running():
            sim.tick()
            #c.data.qpos[2] = 0.5
            #print(c.data.ctrl[0], c.data.ctrl[1])
            #print(controller.problem._der_inverse_ackermann_steering(c.data.ctrl[0], c.data.ctrl[3]))
            #print(c.state)