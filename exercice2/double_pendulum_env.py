import typing as tp

import numpy as np
from gym import spaces
from gym_jiminy.common.envs import BaseJiminyEnv
from gym_jiminy.common.utils import DataNested, sample
from jiminy_py.robot import BaseJiminyRobot
from jiminy_py.simulator import Simulator

SIMULATION_END_TIME = 10.0
STEP_DT = 0.005

class DoublePendulumEnv(BaseJiminyEnv):
    """Implementation of a Gym environment for the DoublePendulum which is using
    Jiminy Engine to perform physics computations and Meshcat for rendering."""

    def __init__(self, naive : bool = True, debug: bool = False) -> None:
        """
        """
        # Get URDF path
        urdf_path = "double_pendulum.urdf"
        hardware_path = "double_pendulum_hardware.toml"

        # Instantiate robot
        robot = BaseJiminyRobot()
        robot.initialize(urdf_path, hardware_path, has_freeflyer=False)

        # Instantiate simulator
        simulator = Simulator(robot)
        simulator.import_options("simulation_config.toml")

        # Configure the learning environment
        super().__init__(simulator, step_dt=STEP_DT, debug=debug)

        self.naive = naive

    def _setup(self) -> None:
        """
        Called by the reset method.
        """
        # Call base implementation
        super()._setup()

    def _initialize_observation_space(self) -> None:
        """Configure the observation space of the environment.
        """
        # Compute observation bounds
        high = np.array([np.pi,
                         np.pi,
                         *self.robot.velocity_limit])

        # Set the observation space
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float64)

    def _initialize_action_space(self) -> None:
        """ Configure the action space of the environment.
        """
        # Define action space to be the target torque
        self.action_space = spaces.Box(
            low=-self.robot.command_limit[self.robot.motors_velocity_idx],
            high=self.robot.command_limit[self.robot.motors_velocity_idx],
            dtype=np.float64
        )

    def _sample_state(self) -> tp.Tuple[np.ndarray, np.ndarray]:
        """ Sample the initial state of robot.
        """
        qpos = sample(scale=np.array([
            np.pi, np.pi]), rg=self.rg)
        qvel = sample(scale=np.array([
            -2, 2]), rg=self.rg)
        return qpos, qvel

    def refresh_observation(self) -> None:
        """Compute the observation based on the current state of the robot."""
        if self.naive:
            position = self.system_state.q
            velocity = self.system_state.v
        else:
            position = self.robot.sensors_data['EncoderSensor'][0] 
            velocity = self.robot.sensors_data['EncoderSensor'][1] 
        self._observation[0:2] = np.mod(position + np.pi, 2*np.pi) - np.pi
        self._observation[2:4] = velocity

    def is_done(self) -> bool:
        """ Return true if episode has met an ending criteria
        """
        if self.simulator.stepper_state.t > SIMULATION_END_TIME:
            return True
        return False

    def compute_command(self,
                        measure: DataNested,
                        action: np.ndarray) -> np.ndarray:
        """Compute the motors efforts to apply on the robot.

        :param measure: Observation of the environment.
        :return: Desired motors efforts.
        """
        # # Call base implementation
        action = np.clip(action, -0.5*self.robot.command_limit, 0.5*self.robot.command_limit)
        action = super().compute_command(measure, action)
        return action

    def compute_reward(self,  # type: ignore[override]
                       *, info: tp.Dict[str, tp.Any]) -> float:
        """ Compute reward 
        """
        # pylint: disable=arguments-differ

        reward = 0.0
        if not self._num_steps_beyond_done:  # True for both None and 0
            reward += 1.0
        return reward

    def step(
        self, action: tp.Optional[DataNested] = None
    ) -> tp.Tuple[DataNested, float, bool, tp.Dict[str, tp.Any]]:

        # Call base implementation
        obs, reward, done, info = super().step(action)

        return obs, reward, done, info