from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'

#used for install and read all the libaries from requirements.txt
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj: #open each line of requirements.txt
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]


        #run and install the requirements.txt without -e .(ignore it)
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        return requirements


setup(
    name='DiamondPriceprediction',
    version='0.0.1',
    author='Pwskills',
    author_email="example@gmail.com",
    install_requires=get_requirements('requirements.txt'), #state which libraries want to install
    packages=find_packages() #find the sub modules
)