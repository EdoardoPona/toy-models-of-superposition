# installing the package with pip install -e . will make the package available in your current environment

from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as req:
        content = req.read()
        requirements = content.split('\n')
    return requirements


setup(
    name='toy_models_of_superposition',
    version='0.1',
    packages=find_packages(),
    install_requires=read_requirements(),
    url='https://github.com/EdoardoPona/toy-models-of-superposition',
    author='Edoardo Pona',
    author_email='edoardo.pona@gmail.com',
    description='utils for training toy models to study superposition', 
)
