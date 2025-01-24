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
import mujoco as mj

def quaternion_from_z_rotation(rotation_z):

    w = math.cos(rotation_z / 2)
    x = 0
    y = 0
    z = math.sin(rotation_z / 2)
    
    return f"{w} {x} {y} {z}"

phi0 = 3.14*3/4
car_pos = np.array([-1.5, 1.5, 0.05])
car_quat = quaternion_from_z_rotation(phi0)
path_points = np.array(
    [
        #[0, 0],
        #[1.5, 1.5],
        #[2, 2],
        #[3, 2],
        #[4, 1],
        #[4.5, 0],
        #[4, -1],
        #[3, -2],
        #[2, -2],
        #[1, -1],
        #[0, 0],
        [-1.5, 1.5],
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


    c = Car_Control_Model(has_trailer=False)

    scn.add_object(c, pos=f"{car_pos[0]} {car_pos[1]} {car_pos[2]}", quat=car_quat)

    m = Trajectory_Marker(x = path_points[:, 0], y = path_points[:,1])
    params = load_mpcc_params()
    #c.controller = control  
    c.trajectory = traj
    scn.add_object(m)
    sim = simulator.Simulator(scn)

    control_model, control_data , xml_name = create_control_model(car_pos, car_quat)

    #controller = F1TENTHMJMPC_SQP(control_model, control_data, trajectory=traj, params=params, xml_path= xml_name)
    sim.model.opt.timestep = 0.01
    c.CTRL_FREQ = 20
    #c.controller = controller
   
    qpos0 = np.zeros(c.model.nq)

    qpos0[:3] = car_pos
    qpos0[3] = 3.14*3/4
    i = 0
    change = 0.5

    forcetorque = np.zeros(6)
    max_force = 0
    c.model.opt.enableflags = 1
    print(c.model.opt)
    with sim.launch():
        mj.mju_copy(c.data.qpos, qpos0)
        while sim.viewer.is_running():
            sim.tick()
            c.set_steer_angles(change)
            delta = c.data.qpos[6]
            print(delta)
            for j,k in enumerate(c.data.contact):
                mj.mj_contactForce(c.model, c.data, j, forcetorque)
                #print(forcetorque)
                max_force = max(max_force, *forcetorque)
                #print(forcetorque)
            #print(max_force)
            #print(f"current delta: {i}")
            if delta >= 0.6 or delta <= -0.6:
                change = -change

            

