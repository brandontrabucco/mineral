"""Author: Brandon Trabucco, Copyright 2019"""


from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    'tensorflow-gpu==2.0.0a0', 
    'numpy', 
    'matplotlib', ]


setup(name='jetpack', version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('jetpack')],
    description='Reinforcement Learning Jetpack',)