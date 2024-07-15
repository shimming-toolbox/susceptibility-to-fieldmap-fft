from setuptools import find_packages, setup
from os import path

# Get the directory where this current file is saved
here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="susceptibility-to-fieldmap-fft",
    python_requires=">=3.8.8",
    description="Code to compute the magnetic field variation from a given susceptibility distribution placed in a constant magnetic field.",
    long_description=long_description,
    url="https://github.com/shimming-toolbox/susceptibility-to-fieldmap-fft",
    entry_points={
        'console_scripts': [
            "compute_fieldmap=fft_simulation.fft_simulation:compute_fieldmap",
            "analytical_cases=fft_simulation.analytical_cases:compare_to_analytical"
        ]
    },
    packages=find_packages(exclude=["docs"]),
    install_requires=[
        "click",
        "numpy>=1.24.4",
        "nibabel>=5.2.1",
        "matplotlib>=3.7.5",
        "scipy>=1.10.1"
    ],
)
