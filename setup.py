import os

from setuptools import setup, find_packages

_current_path = os.path.dirname(os.path.realpath(__file__))
_src_path = os.path.join(_current_path, "src")

setup(
    name='CIL',
    version='1.0',
    packages=find_packages(where=_src_path),
    package_dir={'': _src_path},
    url='https://github.com/supernlogn/Computational-Intelligence-Lab',
    license='',
    author='',
    author_email='',
    description='',
    python_requires='>=3.7'
)
