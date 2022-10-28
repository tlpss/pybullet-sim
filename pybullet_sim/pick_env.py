import copy
import dataclasses
import logging
import random
from typing import List

import gym
import numpy as np
import pybullet as p
import pybullet_data
import tqdm
from gym import spaces

from pybullet_sim.assets.path import get_asset_root_folder
from pybullet_sim.hardware.gripper import WSG50
from pybullet_sim.hardware.ur3e import UR3e
from pybullet_sim.hardware.zed2i import Zed2i
from pybullet_sim.utils.demonstrations import Demonstration, save_visual_demonstrations
from pybullet_sim.utils.pybullet_utils import disable_debug_rendering, enable_debug_rendering, get_pybullet_mode

ASSET_PATH = get_asset_root_folder()

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ObjectConfig:

    n_objects = 5
    colors = [(1, 1, 1, 1), (1, 1, 0, 1), (1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (0, 1, 1, 1), (1, 0, 1, 1)]
    object_list = [{"path": str(ASSET_PATH / "cylinder" / "1:2cylinder.urdf"), "scale": (0.03, 0.05)}]


class UR3ePick(gym.Env):
    image_dimensions = (64, 64)
    # names for the different observations when collecting demonstrations
    demonstration_observation_names = ("rgb", "depth")

    initial_eef_pose = [0.4, 0.1, 0.2, 1, 0, 0, 0]  # robot should be out of view.
    pick_workspace_x_range = (
        -0.125,
        0.125,
    )  # all should be reachable by the robot, and should be square (square images)
    pick_workspace_y_range = (-0.45, -0.2)

    def __init__(
        self,
        use_spatial_action_map=False,
        use_motion_primitive=True,
        simulate_realtime=True,
        object_config: ObjectConfig = None,
        pybullet_mode=None,
    ) -> None:
        self.simulate_realtime = simulate_realtime
        self.use_motion_primitive = use_motion_primitive
        self.use_spatial_action_map = use_spatial_action_map
        self.pybullet_mode = pybullet_mode
        if self.pybullet_mode is None:
            self.pybullet_mode = get_pybullet_mode()
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.image_dimensions[0], self.image_dimensions[1], 4), dtype=np.float16
        )
        self.action_space = spaces.Box(
            low=np.array([self.pick_workspace_x_range[0], self.pick_workspace_y_range[0], 0.001, 0]),
            high=np.array([self.pick_workspace_x_range[1], self.pick_workspace_y_range[1], 0.1, np.pi]),
        )

        if not self.use_motion_primitive:
            assert not self.use_spatial_action_map  # can only use spatial action maps with primitive
            raise NotImplementedError
        if object_config is None:
            self.object_config = ObjectConfig()

        # camera on pick workspace (all corners should be reachable)
        # make camera high and very small fov, to approximate an orthographic view (Andy Zeng uses orthographic reprojection through point cloud)
        self.camera = Zed2i(
            [-0.0, -0.3251, 1.0],
            vertical_fov_degrees=14,
            image_size=UR3ePick.image_dimensions,
            target_position=[-0.0, -0.325, 0],
        )

        self.current_episode_duration = 0
        self.max_episode_duration = 2 * self.object_config.n_objects

        super().__init__()

    def reset(self):
        # bookkeeping (should be in the "Task" as it is about the logic of the MDP)
        self.current_episode_duration = 0

        # creation of the environment
        if p.isConnected():
            self.robot.reset(UR3ePick.initial_eef_pose)

            # remove all current objects
            for id in self.all_object_ids:
                p.removeBody(id)

        else:
            # initialize pybullet
            p.connect(self.pybullet_mode)  # or p.DIRECT for non-graphical version
            disable_debug_rendering()  # will do nothing if not enabled.

            p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
            p.setGravity(0, 0, -2)  # makes life easier..
            self._setup_debug_visualisation()

            self.plane_id = p.loadURDF("plane.urdf", [0, 0, -1.0])
            p.loadURDF(str(ASSET_PATH / "ur3e_workspace" / "workspace.urdf"), [0, -0.3, -0.01])

            self.table_id = p.loadURDF(str(ASSET_PATH / "ur3e_workspace" / "workspace.urdf"), [0, -0.3, -0.001])
            self.basket_id = p.loadURDF(
                "tray/tray.urdf", [0.4, -0.2, 0.01], [0, 0, 1, 0], globalScaling=0.6, useFixedBase=True
            )
            self.gripper = WSG50(
                simulate_realtime=self.simulate_realtime
            )  # DON'T USE ROBOTIQ! physics are not stable..
            self.robot = UR3e(
                eef_start_pose=UR3ePick.initial_eef_pose,
                gripper=self.gripper,
                simulate_real_time=self.simulate_realtime,
            )

        self.spawn_objects()

        if self.simulate_realtime:
            enable_debug_rendering()

        return self.get_current_observation()

    def _setup_debug_visualisation(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        p.addUserDebugLine(
            [self.pick_workspace_x_range[0], self.pick_workspace_y_range[0], 0.01],
            [self.pick_workspace_x_range[0], self.pick_workspace_y_range[1], 0.01],
        )
        p.addUserDebugLine(
            [self.pick_workspace_x_range[0], self.pick_workspace_y_range[1], 0.01],
            [self.pick_workspace_x_range[1], self.pick_workspace_y_range[1], 0.01],
        )
        p.addUserDebugLine(
            [self.pick_workspace_x_range[1], self.pick_workspace_y_range[0], 0.01],
            [self.pick_workspace_x_range[1], self.pick_workspace_y_range[1], 0.01],
        )
        p.addUserDebugLine(
            [self.pick_workspace_x_range[1], self.pick_workspace_y_range[0], 0.01],
            [self.pick_workspace_x_range[0], self.pick_workspace_y_range[0], 0.01],
        )

        p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])

    def spawn_objects(self):
        self.workspace_object_ids = []
        for i in range(self.object_config.n_objects):
            object_type = random.choice(self.object_config.object_list)
            scale = np.random.uniform(object_type["scale"][0], object_type["scale"][1])
            position = [0, 0, 0.1]
            position[0] = np.random.uniform(UR3ePick.pick_workspace_x_range[0], UR3ePick.pick_workspace_x_range[1])
            position[1] = np.random.uniform(UR3ePick.pick_workspace_y_range[0], UR3ePick.pick_workspace_y_range[1])
            id = p.loadURDF(object_type["path"], position, globalScaling=scale)
            p.changeVisualShape(id, -1, rgbaColor=random.choice(self.object_config.colors))
            p.changeDynamics(id, -1, lateralFriction=1.0)
            self.workspace_object_ids.append(id)
        # create list of all objects for bookkeeping
        self.all_object_ids = copy.deepcopy(self.workspace_object_ids)
        self._wait_untill_all_objects_stop_moving()

    def step(self, action: np.ndarray):
        self.current_episode_duration += 1

        if self.use_motion_primitive:
            # convert (u,v,theta) to (x,y,z,theta)
            if self.use_spatial_action_map:
                position = self._image_coords_to_world(
                    int(action[0]), int(action[1]), self.get_current_observation()[..., 3]
                )
                action = np.concatenate([position, np.array([action[2]])])
            self.execute_pick_primitive(action)

        # check if object was grasped
        success = self._is_grasp_succesfull()
        reward = self._reward()  # before place!

        logger.debug(f"grasp succes = {success}")

        if success:
            # move to bin (only visually pleasing?) and
            # remove object from list.
            lifted_object_id = self._get_lifted_object_id()
            self._drop_in_bin(lifted_object_id)
            self.workspace_object_ids.remove(lifted_object_id)

        # move robot back to initial pose
        if self.simulate_realtime:
            self._move_robot(UR3ePick.initial_eef_pose[:3], speed=0.005)
        else:
            self._move_robot_no_physics(UR3ePick.initial_eef_pose[:3])

        # now wait untill all objects have stopped moving before taking an image
        # slightly unrealistic but it helps to make things Markovian
        # which makes it easier to learn for the policy.

        self._wait_untill_all_objects_stop_moving()
        self._prune_objects_outside_of_workspace()
        done = self._done()  # after bookkeeping!
        new_observation = self.get_current_observation()
        return new_observation, reward, done, {}

    def get_current_observation(self):
        rgb, depth, _ = self.camera.get_image()
        rgb = rgb.astype(np.float32) / 255.0  # range [0,1] as it will become float with depth map
        return np.concatenate([rgb, depth[:, :, np.newaxis]], axis=-1)

    def execute_pick_primitive(self, grasp_pose: np.ndarray):

        grasp_position = grasp_pose[:3]
        grasp_position[2] = max(
            grasp_position[2] - 0.02, 0.01
        )  # position is top of object -> graps 2cm below unless this < 0.01cm.

        grasp_orientation = grasp_pose[3]
        pregrasp_position = np.copy(grasp_position)
        pregrasp_position[2] += 0.15
        try:
            self.gripper.open_gripper()
            if self.simulate_realtime:
                self._move_robot(pregrasp_position, grasp_orientation, speed=0.005)
            else:
                self._move_robot_no_physics(pregrasp_position, grasp_orientation)
            self._move_robot(grasp_position, grasp_orientation, max_steps=500)
            self.gripper.close_gripper(max_force=50)
            self._move_robot(pregrasp_position)

        except Exception as e:
            print(f"COULD NOT EXECUTE PRIMITIVE, exception = {e}")

    def _reward(self) -> float:
        if self.use_motion_primitive:
            return self._is_grasp_succesfull() * 1.0

    def _is_object_outside_of_workspace(self, object_id: int) -> bool:
        position, _ = p.getBasePositionAndOrientation(object_id)
        x, y, z = position

        for coord, range in zip([x, y], [self.pick_workspace_x_range, self.pick_workspace_y_range]):
            if coord < range[0] or coord > range[1]:
                return True
        return False

    def _prune_objects_outside_of_workspace(self):
        for object_id in self.workspace_object_ids:
            if self._is_object_outside_of_workspace(object_id):
                logging.debug("removed an object as it was no longer inside the workspace")
                self.workspace_object_ids.remove(object_id)
                self.all_object_ids.remove(object_id)
                p.removeBody(object_id)

    def _done(self):
        done = len(self.workspace_object_ids) == 0
        done = done or self.current_episode_duration >= self.max_episode_duration
        return done

    def _move_robot(self, position: np.array, gripper_z_orientation: float = 0.0, speed=0.001, max_steps=1000):

        eef_target_pose = np.zeros(7)
        eef_target_pose[3:] = p.getQuaternionFromEuler([np.pi, 0.0, gripper_z_orientation])
        eef_target_pose[0:3] = position
        # eef_target_position = self._clip_target_position(eef_target_position)

        logger.debug(f"target EEF pose = {eef_target_pose.tolist()[:3]}")

        self.robot.movep(eef_target_pose, speed=speed, max_steps=max_steps)

    def _move_robot_no_physics(self, position: np.array, gripper_z_orientation: float = 0.0):
        eef_target_pose = np.zeros(7)
        eef_target_pose[3:] = p.getQuaternionFromEuler([np.pi, 0.0, gripper_z_orientation])
        eef_target_pose[0:3] = position
        self.robot.reset(eef_target_pose)
        # make sure the target joint positions are also set correctly
        # otherwise the robot "wants to jump back to its previous position"
        # TODO: this should be handled in robot reset by cancelling all forces and setting targets correctly.
        self._move_robot(position, gripper_z_orientation, max_steps=100)

    def get_oracle_action(self):
        if self.use_motion_primitive:
            return self._oracle_get_pick_pose()

    def render(self, mode=None):
        rgb, _, _ = self.camera.get_image()
        return rgb

    def _oracle_get_pick_pose(self) -> np.ndarray:
        # get heighest object from list
        random_object_id = np.random.choice(np.array(self.workspace_object_ids))

        # get position of that object
        heighest_object_position = p.getBasePositionAndOrientation(random_object_id)[0]
        heighest_object_position = np.array(heighest_object_position)
        heighest_object_position[2] -= 0.001  # firmer grasp
        pick_pose = np.concatenate([heighest_object_position, np.zeros((1,))])
        if not self.use_spatial_action_map:
            return pick_pose
        else:
            img_point = np.linalg.inv(self.camera.extrinsics_matrix) @ np.concatenate(
                [heighest_object_position, np.ones((1,))]
            )
            coordinate = self.camera.intrinsics_matrix @ img_point[:3]
            coordinate /= coordinate[2]
            coordinate = np.clip(coordinate, 0, UR3ePick.image_dimensions[0] - 1)  # make sure poses are reachable
            return np.concatenate([coordinate[:2].astype(np.uint8), np.zeros((1,))])

    def _is_grasp_succesfull(self):
        return (
            self.gripper.get_relative_position() < 0.95
            and max(self._get_object_heights(self.workspace_object_ids)) > 0.1
        )

    def _get_lifted_object_id(self):
        """heuristic for lifted object ID -> get the object that is heighest at that moment (assumes the gripper is lifted)"""
        assert self._is_grasp_succesfull()
        heightest_object_index = np.argmax(np.array(self._get_object_heights(self.workspace_object_ids)))
        return self.workspace_object_ids[heightest_object_index]

    def _drop_in_bin(self, object_id):
        bin_posisition = p.getBasePositionAndOrientation(self.basket_id)[0]
        drop_position = np.array(bin_posisition) + np.array([0, 0, 0.15])

        if self.simulate_realtime:
            self._move_robot(drop_position)
            self.gripper.open_gripper()
        else:
            # just drop object in bin w/o moving robot, which will then be reset anyways.
            p.resetBasePositionAndOrientation(object_id, drop_position, [0, 0, 0, 1])

    def _image_coords_to_world(self, u: int, v: int, depth_map: np.ndarray) -> np.ndarray:
        img_coords = np.array([u, v, 1.0])
        ray_in_camera_frame = np.linalg.inv(self.camera.intrinsics_matrix) @ img_coords
        z_in_camera_frame = depth_map[v, u]  # Notice order!!
        t = z_in_camera_frame / ray_in_camera_frame[2]
        position_in_camera_frame = t * ray_in_camera_frame

        position_in_world_frame = (
            self.camera.extrinsics_matrix @ np.concatenate([position_in_camera_frame, np.ones((1,))])
        )[:3]
        return position_in_world_frame

    @staticmethod
    def _get_object_heights(object_ids: List) -> List[float]:
        heights = []
        for id in object_ids:
            state = p.getBasePositionAndOrientation(id)
            heights.append(state[0][2])
        return heights

    def _is_any_object_moving(self) -> bool:
        for object in self.workspace_object_ids:
            lin_vel, angular_vel = p.getBaseVelocity(object)
            if np.linalg.norm(np.array(list(lin_vel))) > 0.05 or np.linalg.norm(np.array(list(angular_vel))) > 0.01:
                return True
        return False

    def _wait_untill_all_objects_stop_moving(self):
        # make sure objects get a chance to move
        for _ in range(30):
            p.stepSimulation()
        # keep stepping untill all have stopped moving
        while self._is_any_object_moving():
            p.stepSimulation()
            if self.simulate_realtime:
                pass
                # not simulate this in realtime to increase speed.
                # might result in some strange behaviour but does not influence
                # the robot interactions..
                # time.sleep(1/240)

    def collect_demonstrations(self, n_demonstrations: int, path: str):
        """
        Collects and stores demonstrations of the task using the oracle policy. The observation and action type
        are taken from the class instance. Demonstrations are stored as a pickle of a  List of ur-sim::Demonstration's


        :param n_demonstrations:
        :param path: path to store the pickle file
        :return: List of demonstrations
        """
        # TODO: this is a copy from other env. should be moved to a base Env to avoid duplication.
        def store_observation(demonstration: Demonstration, obs):
            demonstration.images[UR3ePick.demonstration_observation_names[0]].append(obs[..., :3])
            demonstration.images[UR3ePick.demonstration_observation_names[1]].append(obs[..., 3])

        demonstrations = []
        for i in tqdm.trange(n_demonstrations):
            demonstration = Demonstration()
            obs = self.reset()
            done = False
            store_observation(demonstration, obs)
            while not done:
                action = self.get_oracle_action()
                obs, reward, done, _ = self.step(action)
                demonstration.actions.append(action)
                demonstration.rewards.append(reward)
                demonstration.dones.append(done)
                store_observation(demonstration, obs)
            demonstrations.append(demonstration)

        save_visual_demonstrations(demonstrations, path)
        return demonstrations


if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.DEBUG)

    env = UR3ePick(simulate_realtime=False, pybullet_mode=p.DIRECT)
    env.collect_demonstrations(2, "pick_env_spatial_actions")
    while True:
        start_time = time.time()
        obs = env.reset()
        post_reset_time = time.time()
        done = False
        while not done:
            img = obs[:, :, :3]
            # plt.imshow(obs[:, :, :3])
            # plt.show()
            # u = int(input("u"))
            # v = int(input("v"))
            # print(obs[u, v, -1])
            # position = env._image_coords_to_world(u, v, obs[:, :, -1])
            # print(position)
            # obs, reward, done, _ = env.step(np.concatenate([position, np.zeros((1,))]))
            obs, reward, done, _ = env.step(env.get_oracle_action())
        done_time = time.time()

        print(f"reset duration = {post_reset_time - start_time}")
        print(f"episode time = {done_time - post_reset_time}")
