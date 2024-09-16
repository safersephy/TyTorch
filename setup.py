# setup.py
from setuptools import setup, find_packages

setup(
    name="Tytorch",  # Name of the package
    version="0.3.0",  # Package version
    author="Tijs van der Velden",
    author_email="tijsvdvelden@hotmail.com",
    description="A straightforward package to simplify pytorch training",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/safersephy/tytorch",  # URL of your GitHub repository
    packages=find_packages(),  # Automatically find package directories
    install_requires=[  # External dependencies
        # Add required packages here
        # e.g., 'numpy', 'torch', etc.
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Minimum Python version requirement
)
