from setuptools import find_packages, setup
from typing import List
import os

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    # Make sure we use the directory of setup.py as base
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, file_path)
    
    if not os.path.exists(full_path):
        return []

    with open(full_path, encoding='utf-8') as file_obj:
        requirements = file_obj.readlines()
        # Strip whitespace and newlines
        requirements = [req.strip() for req in requirements]
        
        # Remove empty lines and comments
        requirements = [req for req in requirements if req and not req.startswith('#')]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name='src', # Naming it 'src' or the project name
    version='0.0.1',
    author='Rahul Kumar pradhan',
    author_email='contactrkp21@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
