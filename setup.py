# setup.py
from setuptools import setup, find_packages

setup(
    name='pyQuantTools',
    version='0.1.0',
    description='A library of tools for quantitative financial research.',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/larrykann/pyQuantTools',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        # Add other dependencies as needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Choose your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

