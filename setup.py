#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='survLime',
      version='0.1',
      description='Survival adaptation of the LIME algorithm',
      package_dir={'': '.'},
      packages=find_packages(),
      install_requires=[],
      test_suite="tests",
      include_package_data=True
)
