import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'rl_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', ['resource/env.pkl']),
        ('share/' + package_name + '/resource', ['resource/model.zip']),

        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=[
        'setuptools',
        'gymnasium',
        'gymnasium-robotics',
        'stable-baselines3'
        ],
    zip_safe=True,
    maintainer='alijnadi',
    maintainer_email='a.jnadi@innopolis.university',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'env_node   = rl_control.env_node:main',
            'agent_node = rl_control.agent_node:main',
        ],
    },
)