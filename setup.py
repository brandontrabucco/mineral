"""Author: Brandon Trabucco, Copyright 2019"""


from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    'cffi',
    'numpy',
    'matplotlib',
    'mujoco-py',
    'tensorflow-gpu==2.0.0b1',
    'gym[all]']


setup(
    name='mineral', version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('mineral')],
    description='A minimalist reinforcement learning package for TensorFlow 2.0')
