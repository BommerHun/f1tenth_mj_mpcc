import numpy as np
import os
import xml.etree.ElementTree as ET
import mujoco as mj
#from aiml_virtual.simulated_object.dynamic_object.controlled_object.car import Car
from car_model import Car
from aiml_virtual import scene, simulator, xml_directory
from trajectory_util import Trajectory_Marker, Spline_2D
from aiml_virtual.trajectory.car_trajectory import CarTrajectory
from mjpcipopt import F1TENTHMJPC
import math
import yaml
import copy

def quaternion_from_z_rotation(rotation_z):

    w = math.cos(rotation_z / 2)
    x = 0
    y = 0
    z = math.sin(rotation_z / 2)
    
    return f"{w} {x} {y} {z}"

car_pos = np.array([0, 0, 0.05])
car_quat = quaternion_from_z_rotation(3.14/4)
path_points = np.array(
    [
        [0, 0],
        [1.5, 1.5],
        [2, 2],
        [3, 2],
        [4, 1],
        [4.5, 0],
        [4, -1],
        [3, -2],
        [2, -2],
        [1, -1],
        [0, 0],
        [-1, 1],
        [-2, 2],
        [-3, 2],
        [-4, 1],
        [-4.5, 0],
        [-4, -2.1],
        [-3, -2.3],
        [-2, -2],
        [-1, -1],
        [0, 0],
    ]
)


def create_control_model(c_pos, c_quat):
    scn = scene.Scene(os.path.join(xml_directory, "empty_checkerboard.xml"), save_filename=os.path.join("xml_models", "control_scene.xml"))

    c = Car()
    scn.add_object(c, pos=f"{car_pos[0]} {car_pos[1]} {car_pos[2]}", quat=car_quat)
    sim = simulator.Simulator(scn)

    return sim.model, sim.data, scn.xml_name

def load_mpcc_params(filename = "control_params.yaml"):
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

    controller = F1TENTHMJPC(control_model, control_data, trajectory=traj, params=params)
    sim.model.opt.timestep = params["dt"]

    #c.controller = controller
  
    with sim.launch():
        while sim.viewer.is_running():
            sim.tick()

            c.data.qpos[2] = 0.5
            c.data.qvel[5] = 1

            qpos_rel = np.zeros(controller.model.nv)
            c.set_torques(0.2)

            mj.mj_differentiatePos(c.model, qpos_rel, 1, controller.qpos0, c.data.qpos)
            print(qpos_rel[7])
            qpos_actual = controller.qpos0.copy()

            mj.mj_integratePos(c.model, qpos_actual, qpos_rel, 1)
            print(qpos_actual[6])
            print(qpos_actual[7])
            #c.data.qpos[2] = 0.5
            #print(c.data.ctrl[0], c.data.ctrl[1])
            #print(controller.problem._der_inverse_ackermann_steering(c.data.ctrl[0], c.data.ctrl[3]))
            #print(c.state)