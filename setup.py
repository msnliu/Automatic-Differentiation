#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 15:03:49 2021

@author: aditimemani
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(name='ad_AHJZ',
      version='1.9',
      packages= setuptools.find_packages(),
      author="Aditi Memani, Hari Raval, Joseph Zuccarelli, liuzongjun",
      description="A Package that calculates Automatic Differentiation for both Scalar and Vector Inputs",
      #long_description= long_description,
      #long_description_content_type="text/markdown",
      url="https://github.com/cs107-AHJZ/cs107-FinalProject.git",
      classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
      ],
      python_requires='>=3.6',
      install_requires = ['numpy==1.19.3' ]
)
