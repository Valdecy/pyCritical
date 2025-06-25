from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='pycritical',
    version='1.7.0',
    license='GNU',
    author='Valdecy Pereira',
    author_email='valdecy.pereira@gmail.com',
    url='https://github.com/Valdecy/pyCritical',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'scipy'
    ],
    zip_safe=True,
    description='A Python Library for CPM and PERT Methods',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
