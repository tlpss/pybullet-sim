import abc
import dataclasses
import logging
import random
from typing import List, Tuple

import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces

from pybullet_sim.assets.path import get_asset_root_folder
from pybullet_sim.hardware.gripper import WSG50
from pybullet_sim.hardware.tactile_sensor import GripperTactileRaySensorArray, TactileRaySensorConfig
from pybullet_sim.hardware.ur3e import UR3e
from pybullet_sim.hardware.zed2i import Zed2i
from pybullet_sim.utils.pybullet_utils import disable_debug_rendering, enable_debug_rendering, get_pybullet_mode

ASSET_PATH = get_asset_root_folder()

logger = logging.getLogger(__name__)


class ActionSpace(abc.ABC):
    max_step_sizes: np.ndarray = None

    def n_dimensions(self):
        return len(self.max_step_sizes)

    def get_gym_space(self):
        return spaces.Box(-self.max_step_sizes, self.max_step_sizes, dtype=np.float32)


class _4DOF_EEF_ACTION_SPACE(ActionSpace):
    """
    X-Y-Z delta [m], Theta delta (top-down) [radians], gripper position [relative 0: closed, 1: open]
    """

    # 5cm x,y,z; 10 degrees, 1/10th of gripper reach

    max_step_sizes: np.ndarray = np.array([0.05, 0.05, 0.05, 10 / 180 * np.pi, 0.1])


class ObservationSpace(abc.ABC):
    def get_gym_space(self):
        raise NotImplementedError


_PROPRIO_SENSOR_ARRAY_OBS_SPACE = None


ACTION_SPACES = _4DOF_EEF_ACTION_SPACE
OBSERVATION_SPACES = _PROPRIO_SENSOR_ARRAY_OBS_SPACE


@dataclasses.dataclass
class ObjectConfig:
    path: str = None
    scale_min: float = 0.01
    scale_max: float = 1.5


### Sensor configurations for WSG50

finger_box_dims = (0.013, 0.02, 0.08)
x, y, z = finger_box_dims

epsilon = 0.0001
center_front = TactileRaySensorConfig(np.array([x / 2 + epsilon, 0, z / 2 - 0.005]), np.array([1, 0, 0]))
center_back = TactileRaySensorConfig(np.array([-x / 2 - epsilon, 0, z / 2 - 0.005]), np.array([-1, 0, 0]))
center_side_left = TactileRaySensorConfig(np.array([0, y / 2 + epsilon, z / 2 - 0.005]), np.array([0, 1, 0]))
center_side_right = TactileRaySensorConfig(np.array([0, -y / 2 - epsilon, z / 2 - 0.005]), np.array([0, -1, 0]))
center_top = TactileRaySensorConfig(np.array([0, 0, z / 2 + epsilon]), np.array([0, 0, 1]))
####


@dataclasses.dataclass
class TactilePickConfig:
    object_list: List[ObjectConfig] = None
    sensor_config: List[TactileRaySensorConfig] = None
    action_space: ActionSpace = None
    observation_space = None
    max_episode_steps: int = 200

    # robot workspace
    # all poses should be reachable by the robot!
    pick_workspace_x_range = (-0.2, 0.2)
    pick_workspace_y_range = (-0.45, -0.20)
    workspace_z_range = (0.01, 0.2)
    workspace_euler_x_range = (np.pi / 2, 3 * np.pi)
    workspace_euler_y_range = (0, np.pi)
    workspace_euler_z_range = (0, np.pi)

    # object spawn space
    object_x_space = (0.0, 0.01)
    object_y_space = (-0.2, -0.21)
    object_theta_space = (0.0, 0.01)


DEFAULT_CONFIG = TactilePickConfig(
    object_list=[ObjectConfig(str(ASSET_PATH / "cylinder" / "1:2cylinder.urdf"), 0.03, 0.05)],
    action_space=_4DOF_EEF_ACTION_SPACE(),
    sensor_config=[center_front, center_back, center_side_left, center_side_right, center_top],
)


class TactilePick(gym.Env):
    image_dimensions = (128, 128)

    def __init__(
        self,
        use_spatial_action_map=False,
        use_motion_primitive=True,
        simulate_realtime=True,
        config: TactilePickConfig = DEFAULT_CONFIG,
        pybullet_mode=None,
    ) -> None:
        self.simulate_realtime = simulate_realtime
        self.use_motion_primitive = use_motion_primitive
        self.use_spatial_action_map = use_spatial_action_map
        self.pybullet_mode = pybullet_mode
        if self.pybullet_mode is None:
            self.pybullet_mode = get_pybullet_mode()

        self.action_space = config.action_space.get_gym_space()
        # TODO: fix observation space

        # camera on pick workspace for rendering
        # make camera high and very small fov, to approximate an orthographic view (Andy Zeng uses orthographic reprojection through point cloud)
        self.camera = Zed2i(
            [-0.0, -1.0, 0.7],
            vertical_fov_degrees=40,
            image_size=TactilePick.image_dimensions,
            target_position=[-0.0, -0.325, 0.2],
        )
        self.config = config
        self.current_episode_duration = 0

        super().__init__()

    def reset(self):
        # bookkeeping (should be in the "Task" as it is about the logic of the MDP)
        self.current_episode_duration = 0

        self.robot_state = np.zeros(7)
        self.old_robot_state = np.copy(self.robot_state)

        # creation of the environment
        if p.isConnected():
            p.resetSimulation()

        else:
            p.connect(self.pybullet_mode)  # or p.DIRECT for non-graphical version

        # just create them from scratch each time, somehow this is faster than not resetting the simulator...
        disable_debug_rendering()  # will do nothing if not enabled.

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -2)  # makes life easier..
        self._setup_debug_visualisation()

        self.plane_id = p.loadURDF("plane.urdf", [0, 0, -1.0])
        p.loadURDF(str(ASSET_PATH / "ur3e_workspace" / "workspace.urdf"), [0, -0.3, -0.01])

        self.table_id = p.loadURDF(str(ASSET_PATH / "ur3e_workspace" / "workspace.urdf"), [0, -0.3, -0.001])

        # create random object at random position
        self.object_id, object_position = self.spawn_object()

        # TODO: sample robot pose (avoid cube around obstacle?)
        self.robot_state[:6] = np.array([0.0, -0.3, 0.2, np.pi, 0, 0])

        eef_start_pose = np.zeros(7)
        eef_start_pose[:3] = self.robot_state[:3]
        eef_start_pose[3:] = p.getQuaternionFromEuler(self.robot_state[3:6])
        self.gripper = WSG50(simulate_realtime=self.simulate_realtime)  # DON'T USE ROBOTIQ! physics are not stable..

        self.robot = UR3e(
            eef_start_pose=eef_start_pose,
            gripper=self.gripper,
            simulate_real_time=self.simulate_realtime,
        )

        self.tactile_sensors = GripperTactileRaySensorArray(
            self.config.sensor_config,
            [
                self.gripper.get_left_finger_position_and_orientation,
                self.gripper.get_right_finger_position_and_orientation,
            ],
        )

        if self.simulate_realtime:
            enable_debug_rendering()

        return self.get_current_observation()

    def _setup_debug_visualisation(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])

    def spawn_object(self) -> Tuple[int, np.ndarray]:
        self.workspace_object_ids = []
        object_config = random.choice(self.config.object_list)
        scale = np.random.uniform(object_config.scale_min, object_config.scale_max)
        position = np.array([0, 0, 0.1])
        position[0] = np.random.uniform(self.config.object_x_space[0], self.config.object_x_space[1])
        position[1] = np.random.uniform(self.config.object_y_space[0], self.config.object_y_space[1])
        theta = np.random.uniform(self.config.object_theta_space[0], self.config.object_theta_space[1])
        orientation = p.getQuaternionFromEuler([0, 0, theta])
        # maximal coordinates for improved performance?
        id = p.loadURDF(object_config.path, position, orientation, useMaximalCoordinates=True, globalScaling=scale)

        # wait for object to settle (and fall on table)
        for _ in range(100):
            p.stepSimulation()
        position = p.getBasePositionAndOrientation(id)[0]
        return id, position

    def step(self, action: np.ndarray):
        self.current_episode_duration += 1

        assert len(action) == self.config.action_space.n_dimensions()
        self.old_robot_state = self.robot_state

        if isinstance(self.config.action_space, _4DOF_EEF_ACTION_SPACE):
            delta_state = np.zeros(7)
            delta_state[:3] = action[:3]  # position
            delta_state[5] = action[3]  # z-rotation
            delta_state[6] = action[4]  # gripper
        else:
            delta_state = action

        self.robot_state += delta_state
        self.robot_state = self._clip_robot_state(self.robot_state)

        self._move_to_new_robot_state(self.robot_state)

        # check if object was grasped
        self._is_grasp_succesfull()
        reward = self._reward()
        done = self._done()
        new_observation = self.get_current_observation()
        return new_observation, reward, done, {}

    def _clip_robot_state(self, state: np.ndarray):
        assert len(state) == 7  # 6DOF + gripper.
        ranges = [
            self.config.pick_workspace_x_range,
            self.config.pick_workspace_y_range,
            self.config.workspace_z_range,
            self.config.workspace_euler_x_range,
            self.config.workspace_euler_y_range,
            self.config.workspace_euler_z_range,
            (0, 1),
        ]

        for index, range in enumerate(ranges):
            state[index] = np.clip(state[index], range[0], range[1])
        return state

    def _move_to_new_robot_state(self, state: np.ndarray):
        assert len(state) == 7
        self.gripper.movej(state[-1], max_steps=10)

        pose = np.zeros(7)
        pose[:3] = state[:3]
        pose[3:] = p.getQuaternionFromEuler(state[3:6])
        logger.debug(f"target EEF pose = {pose.tolist()}")
        self.robot.movep(pose, max_steps=10, speed=0.001)

    def get_current_observation(self):
        return None

    def _reward(self) -> float:
        return None

    def _is_object_outside_of_workspace(self) -> bool:
        position, _ = p.getBasePositionAndOrientation(self.object_id)
        x, y, z = position

        for coord, range in zip([x, y], [self.config.object_x_space, self.config.object_y_space]):
            if coord < range[0] or coord > range[1]:
                return True
        return False

    def _done(self):
        done = self._is_grasp_succesfull()
        done = done or self.current_episode_duration >= self.config.max_episode_steps
        return done

    def render(self, mode=None):
        rgb, _, _ = self.camera.get_image()
        return rgb

    def _is_grasp_succesfull(self):
        # TODO: check if object was lifted
        # requires to have the base position of the object before it was lifted
        # and check if it is now x cm higher (and enclosed in the gripper to avoid the object being pushed up?
        return False


if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.DEBUG)

    env = TactilePick(simulate_realtime=True, pybullet_mode=p.GUI)
    env.reset()
    env.robot.movep([-0.01, -0.2, 0.01, 1.0, 0, 0, 0])
    env.tactile_sensors.visualize_sensors()
    print(env.tactile_sensors.get_sensor_distances())
    time.sleep(100)
    while True:
        start_time = time.time()
        obs = env.reset()
        post_reset_time = time.time()
        done = False
        while not done:
            obs, reward, done, _ = env.step(env.action_space.sample())
            # img = env.render()
            time.sleep(1 / 240)
        done_time = time.time()

        print(f"reset duration = {post_reset_time - start_time}")
        print(f"episode time = {done_time - post_reset_time}")
