from setuptools import setup, find_packages

setup(
    name="pystarshade",
    version="0.1.8",
    description="A python package for starshade imaging simulations",
    author="Jamila Taaki",
    author_email="xiaziyna@gmail.com",
    url="https://github.com/xiaziyna/PyStarshade",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "hcipy",
        "astropy",
        "setuptools",
        "pytest"
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
)
