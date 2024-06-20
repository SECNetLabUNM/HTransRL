from setuptools import setup

setup(
    name="air-corridor",
    version="0.0.1",
)

# g
'''
gym_examples is package name, can be found in "pip list"
When the package name defined in the setup.py file is different from the actual directory name 
where your package's source code resides, 
you need to import the modules using the actual package name.

here the acutual package name is 'myenv'
in code:
import myenv
'''