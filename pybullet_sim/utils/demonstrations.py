import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import imageio
import numpy as np

"""
data collection inspired by https://github.com/stepjam/RLBench/blob/085ce8671e01841c0d2355a76110fc51c5e9c9aa/rlbench/task_environment.py#L110
"""


@dataclass
class Demonstration:
    """
    A very simple data container for collecting first person demonstrations
    """

    # must use the default factory! if you simply create
    # observations = [], this becomes a class attribute and is hence
    # shared amongst all instances..
    images: Dict[str, List[np.ndarray]] = field(default_factory=lambda: defaultdict(lambda: []))
    states: List[np.ndarray] = field(default_factory=lambda: [])
    actions: List[np.ndarray] = field(default_factory=lambda: [])
    rewards: List[np.ndarray] = field(default_factory=lambda: [])
    dones: List[np.ndarray] = field(default_factory=lambda: [])


def save_visual_demonstrations(demonstrations: List[Demonstration], path):
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    for i, demo in enumerate(demonstrations):
        demonstration_path = path / f"{i}"
        demonstration_path.mkdir()
        for name, observation in demo.images.items():
            demonstration_obs_path = demonstration_path / name
            demonstration_obs_path.mkdir()
            for j, obs in enumerate(observation):
                imageio.imwrite(demonstration_obs_path / f"{j}.png", obs)

        # don't store these images in pickle
        demo.images = None
        with open(str(demonstration_path / "demonstration.pkl"), "wb") as file:
            pickle.dump(demo, file)


if __name__ == "__main__":
    a = Demonstration()
    b = Demonstration()
    a.images["rgb"] = np.zeros((64, 64, 3))
    print(b.images)
    print(a.images)
