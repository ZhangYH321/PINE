from setuptools import find_packages, setup


setup(
    name="pine-airsim-env",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["gym", "airsim", "numpy", "opencv-python", "PyQt5"],
)
