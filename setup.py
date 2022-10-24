import setuptools

setuptools.setup(
    name="pybullet-sim",
    version="0.0.1",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    description="robot manipulation simulation environments in pybullet",
    packages=["pybullet_sim"],
    install_requires=[
        # requirements.txt used as it is not possible -e install?
        #  and ur_ikfast requires this.
        "ur_ikfast",
    ],
)
