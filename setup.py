from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup
from setuptools import find_packages

def _requires_from_file(filename):
    return open(filename).read().splitlines()

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="KernelMethod",
    version="1.0.0",
    license="MIT License",
    description="A Python Package for Kernel methods.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Yu Kiyokawa",
    author_email='dummn.marionette.7surspecies@gmail.com',
    url="https://github.com/JohnYKiyo/kernel_method.git",
    keywords='kernel, Kernel ABC, Kernel Mean',
    python_requires=">=3.6.0",
    packages = [s.replace('kernelmtd','kernelmtd') for s in find_packages('.')],
    package_dir={"kernelmtd": "kernelmtd"},
    package_data={'kernelmtd': ['data/*.npy', 'data/*.csv']},
    py_modules=[splitext(basename(path))[0] for path in glob('kernelmtd/*.py')],
    install_requires=_requires_from_file('requirements.txt'),
)