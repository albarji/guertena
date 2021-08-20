from setuptools import setup


setup(
    name="guertena",
    version="0.0.1",
    description="Guertena is an easy to use, quality oriented python library for neural style transfer.",
    long_description="""
Guertena is an easy to use, quality oriented python library for neural style transfer.
""",
    packages=['guertena'],
    install_requires=[
        'cudatoolkit',
        'numpy=1.20.*',
        'pytorch=1.9.*',
        'torchvision=0.10.*'
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
    keywords='neural style transfer',
    test_suite="pytest",
)