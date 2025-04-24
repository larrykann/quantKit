"""Setup script for quantKit.

Defines package metadata and dependencies for installation via setuptools.
"""
from setuptools import setup, find_packages

setup(
    name='quantKit',
    version='0.1.0',
    description='A library of tools for quantitative financial research.',
    author='Larry Kann',
    author_email='larry@huntgathertrade.com',
    url='https://github.com/larrykann/quantKit',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'rich',
        'numba',
        # Add other dependencies as needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Choose your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

