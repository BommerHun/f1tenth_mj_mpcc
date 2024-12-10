import numpy as np
import os
import xml.etree.ElementTree as ET
import mujoco as mj
#from aiml_virtual.simulated_object.dynamic_object.controlled_object.car import Car
from car_model import Car
from aiml_virtual import scene, simulator, xml_directory
from trajectory_util import Trajectory_Marker, Spline_2D
from mjpc import f1tenth_mjpc
from aiml_virtual.trajectory.car_trajectory import CarTrajectory
from mjpcipopt import OptProblem
import math
from mjpc import f1tenth_mjpc
car_pos = np.array([0, 0, 0.05])
car_quat = "1 0 0 0"
path_points = np.array(
    [
        [0, 0],
        [1, 1],
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
    scn.add_object(c, " ".join(map(str, c_pos + np.array([0, 0, 1]))), " ".join(map(str, c_quat)), "0.5 0.5 0.5 1")

    sim = simulator.Simulator(scn)

    return sim.model, sim.data, scn.xml_name

if __name__ == "__main__":
    scn = scene.Scene(os.path.join(xml_directory, "empty_checkerboard.xml"), save_filename=os.path.join("xml_models", "main_scene.xml"))

    traj = CarTrajectory()
    traj.build_from_points_const_speed(path_points, path_smoothing=0.01, path_degree=4, const_speed=1.5)

    c_model, c_data, c_scene = create_control_model(c_pos= car_pos, c_quat=car_quat)

    c = Car(has_trailer=False)

    scn.add_object(c, pos="0 0 0.052", quat='0.9485664043524404 0 0 0.31657823130133655')

    m = Trajectory_Marker(x = path_points[:, 0], y = path_points[:,1])

    #c.controller = control  
    c.trajectory = traj
    scn.add_object(m)
    sim = simulator.Simulator(scn)
    op = OptProblem()
    with sim.launch():
        while sim.viewer.is_running():
            sim.tick()
            
            c.set_steer_angles(c.data.ctrl[0]+0.01)

            print(op._inverse_ackerman_steering(c.data.ctrl[0], c.data.ctrl[3]))
            print(op._der_inverse_ackermann_steering(c.data.ctrl[0], c.data.ctrl[3]))
            #c.data.qpos[2] = 0.5
            #c.data.qpos[0] = 0
            #c.data.qpos[1] = 0
            #c.data.qpos[5] = 0
            #c.data.qpos[4] = 0
            #c.data.qpos[6] = 0
            #c.data.ctrl[6] = 0.5
            #c.set_torques(0.05)            

            #print(control._cost_e_c(point, theta))
            #print(control._cost_e_l(point, theta))
            #print(c.data.qpos[13])
            #print(f"nx: {np.shape(c.data.qpos)}\nnx_dot: {np.shape(c.data.qvel)}\nnu:{np.shape(c.data.ctrl)}")