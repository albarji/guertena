from setuptools import setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="guertena",
    version="0.0.1",
    description="Guertena is an easy to use, quality oriented python library for neural style transfer.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['guertena'],
    install_requires=[
        'cudatoolkit>=10',
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