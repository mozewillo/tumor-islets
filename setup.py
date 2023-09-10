#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='tumor_islets',
    version='0.0.1',
    author='immucan-uw',
    packages = [
        'tumor_islets',
        'tumor_islets.graph',
        'tumor_islets.plots',
        'tumor_islets.clustering',
        'tumor_islets.additional',
    ],
    description="Graph-based method for analyzing the IF data",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mozewillo/tumor-islets',
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.8.0',
    install_requires=['scipy>=1.8.1',
                      'pandas>=1.4.3',
                      'matplotlib>=3.5.2',
                      'numpy>=1.22.4',
                      'sklearn',
                      'shapely==1.8.2',
                      'descartes>=1.1.0',
    ])
