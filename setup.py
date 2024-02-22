from setuptools import find_packages,setup
from typing import  List

HYPEN_E_DOT='-e .'
def get_requirnements(file_path:str)->List[str]:
    requirnements=[]
    with open(file_path) as file_obj:
        requirnements=file_obj.readlines()
        requirnements=[req.replace("\n", " ") for req in requirnements]
        if HYPEN_E_DOT in requirnements:
            requirnements.remove(HYPEN_E_DOT)
    return requirnements


setup(
    name='mlproject',
    version='0.0.1',
    author='Yash Shah',
    author_email='yashshah2774@gmail.com',
    packages=find_packages(),
    install_requires=get_requirnements("requirnement.txt")
)