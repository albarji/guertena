import os
from setuptools import setup


# Read long description from readme
with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()


# Get tag from Github environment variables
TAG = os.environ['GITHUB_TAG'] if 'GITHUB_TAG' in os.environ else None


setup(
    name="guertena",
    version=TAG if TAG is not None else "develop",
    description="Guertena is an easy to use, quality oriented python library for neural style transfer.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=['guertena'],
    install_requires=[
        'numpy>=1.20,<2',
        'pytorch>=1.9,<2',
        'torchvision>=0.10,<1'
    ],
    author="Alvaro Barbero",
    url='https://github.com/albarji/guertena',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Environment :: GPU :: NVIDIA CUDA',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Artistic Software',
        'Topic :: Multimedia :: Graphics'
    ],
    keywords='artificial-intelligence, deep-learning, neural-style-transfer',
    test_suite="pytest",
)