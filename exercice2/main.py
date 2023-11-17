import numpy as np
from jiminy_py.simulator import Simulator
from jiminy_py.robot import BaseJiminyRobot

SIMULATION_END_TIME = 10.0
urdf_path = "double_pendulum.urdf"
hardware_path = "double_pendulum_hardware.toml"

robot = BaseJiminyRobot()
robot.initialize(urdf_path, hardware_path, has_freeflyer=False)

simulator = Simulator(robot)
simulator.import_options("simulation_config.toml")

q0 = np.zeros(2)
v0 = np.random.rand(2)
simulator.simulate(SIMULATION_END_TIME, q0, v0)
simulator.replay()
