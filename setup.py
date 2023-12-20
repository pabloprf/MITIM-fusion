from setuptools import setup, find_packages

setup(
    name="MITIM",
    version="1.0.0",
    description="MIT Integrated Modeling Suite for Fusion Applications",
    url="https://mitim-fusion.readthedocs.io/",
    author="P. Rodriguez-Fernandez",
    author_email="pablorf@mit.edu",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pip",
        "numpy",
        "matplotlib",
        "argparse",
        "h5py",
        "netCDF4",
        "xarray",
        "xlsxwriter",
        "statsmodels",
        "dill",
        "ipython",
        "pyDOE",
        "multiprocessing_on_dill",
        "deap",
        "botorch==0.9.4",  # Comes w/ gpytorch==1.11, torch>=1.13.1. PRF also tested w/ torch-2.1.1
        "scikit-image",  # Stricly not for MITIM, but good to have for pygacode
    ],
    extras_require={
        "pyqt": "PyQt6",
        "omfit": [
            "omfit_classes",
            "xarray==2022.3.0",  # As of 12/07/2023, omfit_classes fails for higher versions
            "matplotlib==3.5.3",  # As of 12/07/2023, omfit_classes fails for higher versions
            "omas",
            "fortranformat",
            "openpyxl",
        ],
        "freegs": [
            "Shapely",
            "freegs @ git+https://github.com/bendudson/freegs.git",
        ],
    },
)
