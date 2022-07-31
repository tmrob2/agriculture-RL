from setuptools import setup

setup(
    name='farm_gym',
    version='0.0.1',
    install_requires=[
        'gym', 
        'pcse==5.5.3', 
        'pandas', 
        'PyYAML', 
        'stable-baselines3[extra]',
        'tensorflow'
    ]
)