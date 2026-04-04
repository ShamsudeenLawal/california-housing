from setuptools import setup, find_packages

def get_requirements_from_file(filename):
    with open(filename, 'r') as file:
        requirements = file.readlines()
    # remove empty lines and comments
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]
    requirements = [req.replace('-e .', '') for req in requirements if req.strip() and not req.startswith('#')]
    return requirements

setup(
    name='california_housing',
    version='0.1.0',
    packages=find_packages(),
    install_requires=get_requirements_from_file('requirements.txt'),
    author='Your Name',
    author_email='your.email@example.com',
    description='A Package for California Housing data',
    url='',
)