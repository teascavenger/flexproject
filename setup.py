from setuptools import setup, find_packages

setup(
    name="flextomo",
    package_dir={'flextomo': 'flextomo'},
    packages=find_packages(),

    install_requires=[
    "numpy",
    "astra-toolbox",
    "tqdm",  
    "scipy",
    "flexdata"],

    version='0.0.1',
)