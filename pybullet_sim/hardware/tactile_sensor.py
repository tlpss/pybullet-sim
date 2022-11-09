import dataclasses
import logging
from typing import Callable, List, Tuple

import numpy as np
import pybullet as p
import pybullet_data

from pybullet_sim.assets.path import get_asset_root_folder
from pybullet_sim.hardware.gripper import WSG50
from pybullet_sim.hardware.ur3e import UR3e
from pybullet_sim.utils.transforms import get_homogeneous_matrix_from_position_and_quaternion

ASSET_PATH = get_asset_root_folder()

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TactileRaySensorConfig:
    position_in_finger_frame: np.ndarray = None
    direction_in_finger_frame: np.ndarray = None
    max_distance: float = 0.02


class GripperTactileRaySensorArray:
    def __init__(
        self,
        sensor_config: List[TactileRaySensorConfig],
        finger_pose_funcs: List[Callable[[], Tuple[np.ndarray, np.ndarray]]],
    ) -> None:
        self.pose_funcs = finger_pose_funcs
        self.sensor_config = sensor_config
        self.n_sensors = len(self.sensor_config) * len(self.pose_funcs)
        self.visualisation_ids: List[int] = []

    def _get_transform_matrices_for_each_finger(self) -> List[np.ndarray]:
        matrices = []
        for function in self.pose_funcs:
            position, quaternion = function()
            matrix = get_homogeneous_matrix_from_position_and_quaternion(position, quaternion)
            matrices.append(matrix)
        return matrices

    def _get_ray_start_end_positions(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        ray_starts = []
        ray_ends = []
        transform_matrices = self._get_transform_matrices_for_each_finger()
        for transform_matrix in transform_matrices:
            for sensor in self.sensor_config:
                position = np.ones(4)
                position[:3] = sensor.position_in_finger_frame
                position_in_world_frame = (transform_matrix @ position)[:3]
                direction_in_world_frame = transform_matrix[:3, :3] @ sensor.direction_in_finger_frame
                ray_starts.append(position_in_world_frame)
                ray_ends.append(position_in_world_frame + sensor.max_distance * direction_in_world_frame)
        return ray_starts, ray_ends

    def remove_visualisations(self):
        assert p.isConnected()

        for id in self.visualisation_ids:
            p.removeUserDebugItem(id)

    def visualize_sensors(self):
        assert p.isConnected()
        self.remove_visualisations()
        starts, ends = self._get_ray_start_end_positions()
        relative_distances = self.get_sensor_distances(True)
        for i in range(len(starts)):
            color = [relative_distances[i], (1 - relative_distances[i]), 0]
            point_id = p.addUserDebugPoints([starts[i]], [color], 5)
            line_id = p.addUserDebugLine(starts[i], ends[i], color)
            self.visualisation_ids.extend([point_id, line_id])

    def get_sensor_distances(self, return_relative_distances=False) -> List[float]:
        distances = []
        assert p.isConnected()
        starts, ends = self._get_ray_start_end_positions()
        for start, end in zip(starts, ends):
            fraction = p.rayTest(start, end)[0][2]
            distance = fraction

            if not return_relative_distances:
                distance *= np.linalg.norm(end - start)

            distances.append(distance)
        return distances

    def get_binary_contacts(self) -> List[float]:
        epsilon = 1e-5
        distances = self.get_sensor_distances()
        binary_contacts = [
            1.0 * (distances[i] < self.sensor_config[i].max_distance - epsilon) for i in range(self.n_sensors)
        ]
        return binary_contacts


# TODO: binary contact sensor subclass


if __name__ == "__main__":
    import time

    # wsg
    x, y, z = (0.013, 0.02, 0.08)
    epsilon = 0.0001
    center_front = TactileRaySensorConfig(np.array([x / 2 + epsilon, 0, z / 2 - 0.005]), np.array([1, 0, 0]), 0.015)
    center_top = TactileRaySensorConfig(np.array([0, 0, z / 2 + epsilon]), np.array([0, 0, 1]))

    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -10)
    tableId = p.loadURDF(str(get_asset_root_folder() / "ur3e_workspace" / "workspace.urdf"), [0, -0.3, -0.001])
    disc = p.loadURDF(str(get_asset_root_folder() / "cube.urdf"), [0, -0.2, 0.05], globalScaling=0.05)
    target = p.getDebugVisualizerCamera()[11]
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)

    p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=target)
    gripper1 = WSG50()
    robot1 = UR3e(
        robot_base_position=[0.0, 0.0, 0.0],
        simulate_real_time=True,
        gripper=gripper1,
        eef_start_pose=np.array([0.2, -0.1, 0.3, 1, 0, 0, 0]),
    )
    sensor = GripperTactileRaySensorArray([center_front], [gripper1.get_left_finger_position_and_orientation])
    robot1.movep([0.0, -0.2, 0.01, 1.0, 0, 0, 0])
    sensor.visualize_sensors()
    print(sensor.get_sensor_distances())
    print(sensor.get_binary_contacts())
    s = input("press any key")
    robot1.movep([0.02, -0.2, 0.01, 1.0, 0, 0, 0], speed=0.001)
    sensor.visualize_sensors()
    print(sensor.get_sensor_distances())
    print(sensor.get_binary_contacts())
    for _ in range(20):
        p.stepSimulation()
    print(sensor.get_sensor_distances())
    s = input("press any key")
    robot1.movep([0.03, -0.2, 0.01, 1.0, 0, 0, 0], speed=0.001)
    print(sensor.get_sensor_distances())
    time.sleep(100)
