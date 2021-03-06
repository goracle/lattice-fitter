"""Setup script for latfit package.
"""
# Always prefer setuptools over distutils
from os import path
import codecs
from setuptools import setup, find_packages
# To use a consistent encoding

HERE = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with codecs.open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='latfit',
    version='0.11.0.0',
    description='lattice qcd data fitter',
    long_description=LONG_DESCRIPTION,
    url='https://www.github.com/goracle/lattice-fitter',
    license='GPLv3',
    classifiers=[
        'Development Status :: 4 - Beta'
        'Intended Audience :: Physicists'
        'Topic :: Data Visualisation and Analysis'
        'License :: GPLv3'
        'Programming Language :: Python :: 3.6'
    ],
    keywords='qcd data analysis',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'sympy',
        'scipy',
        'numdifftools',
        'h5py',
        'cloudpickle',
        'gvar',
        'iminuit',
        'accupy',
        'recordtype'
    ],
    entry_points={
        'console_scripts': [
            'latfit = latfit.__main__:main',
        ],
    },
)
