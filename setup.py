"""Author: Brandon Trabucco, Copyright 2019"""


from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    'tensorflow-gpu==2.7.2',
    'cffi',
    'numpy',
    'matplotlib',
    'mujoco-py',
    'gym[all]',]


setup(
    name='mineral', version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('mineral')],
    description='A minimalist reinforcement learning package for TensorFlow 2.0')
