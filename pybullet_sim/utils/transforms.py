from scipy.spatial.transform import Rotation
import pybullet as p
import numpy as np

def get_homogeneous_matrix_from_position_and_quaternion(position:np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """convert a 3D position and 4D quaternion (scalar-last!) to a 4x4 homogeneous matrix
    Args:
        position (np.ndarray): _description_
        quaternion (np.ndarray): _description_
    """

    transform = np.eye(4)
    transform[:3,3] = position
    transform[:3,:3] = Rotation.from_quat(quaternion).as_matrix()
    return transform



if __name__ == "__main__":
    print(get_homogeneous_matrix_from_position_and_quaternion(np.array([1,2,3]),np.array([0,0,0,1])))

