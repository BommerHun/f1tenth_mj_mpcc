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


class OptProblem(ci.Problem):
    def __init__(self, model, data, trajectory, N, qpos_0, weights):
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
        

        #Creating the inverse ackerman formulas:
        # Define variables



    def constraints(self, x):
        """Return constaints

        Args:
            x: optimisation variables
        """
        #This should contain the constraints for the steering, the drivetrain, and the virtual parameter

    
        #TODO: create a symbolic function for evaulating the steering input of the bicycle model from the vehicle's current steering inputs
        #list of constraints: 
        #TODO: Dynamics constraint 1-N \TICK\
        #TODO: Initial condition \TICK\
        #TODO: ack_inv(delta_left)-ack_inv(delta_right) = 0
        #TODO: drivetrain: u_d1 - ud_2 =0, ud_2 - ud_3 = 0, ud_3 - ud_4 = 0
        #TODO: 0 < theta_hat < theta_max
        #TODO: 0 < theta_hat_dot < theta_dot_max
        #TODO: 0 < delta_left, delta_right < delta_max
        #TODO: i.e. ??0?? < u_d1 < u_d_max

        #delta_in_left = cs.atan((-cs.tan(delta_left) * wb) / (0.5 * tw * cs.tan(delta_left) - wb))
        #delta_in_right = cs.atan((-cs.tan(delta_right) * wb) / (0.5 * tw * cs.tan(delta_right) + wb))

        constraints = np.array([])

        k_0 = self.N * self.nx # shows the end of the state trajectory

        """Dynamics Constraints"""
        for k in range(self.N-1):
            constraints = np.append(constraints, x[(k+1)*self.nx:(k+2)*self.nx]-self.dyn_step(x[k*self.nx: (k+1)*self.nx], x[k_0 + k*self.nu : k_0 + (k+1)*self.nu]))
        constraints = np.append(constraints, x[:self.nx] - self.x_0)

        """drivetrain Constraints"""
        #INDEXING OF the actuators: Fr:1 Rl: 2 Fl: 4 Rr: 5
        
        for k in range(self.N-1):
            constraints = np.append(constraints, x[k_0+ k*self.nu +1]- x[k_0+ k*self.nu +2])
            constraints = np.append(constraints, x[k_0+ k*self.nu +2]- x[k_0+ k*self.nu +4])
            constraints = np.append(constraints, x[k_0+ k*self.nu +4]- x[k_0+ k*self.nu +5])
        #Return collected contraints
        return constraints

    def jacobian(self, x, u):
        """Return jacobian of the constraints

        Args:
            x: optimisation variables
        """

        pass

    def dyn_step(self, x):
        pass

    
        