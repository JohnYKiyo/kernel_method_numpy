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
    name="nukernelmtd",
    version="1.0.3",
    license="MIT License",
    description="A Python Package for Kernel methods.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Yu Kiyokawa",
    author_email='dummn.marionette.7surspecies@gmail.com',
    url="https://github.com/JohnYKiyo/kernel_method_numpy.git",
    keywords='kernel, Kernel ABC, Kernel Mean',
    python_requires=">=3.6.0",
    packages = [s.replace('nukernelmtd','nukernelmtd') for s in find_packages('.')],
    package_dir={"nukernelmtd": "nukernelmtd"},
    package_data={'nukernelmtd': ['data/*.npy', 'data/*.csv']},
    py_modules=[splitext(basename(path))[0] for path in glob('nukernelmtd/*.py')],
    install_requires=_requires_from_file('requirements.txt'),
)