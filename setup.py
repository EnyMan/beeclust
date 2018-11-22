from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy


setup(
    name='beeclust',
    version='0.1',
    description='BeeClust simulation',
    author='Martin Pitak',
    author_email='pitakma1@fit.cvut.cz',
    keywords='numpy,console,simulation',
    license='Public Domain',
    url='https://github.com/EnyMan/beeclust',
    #packages=find_packages(),
    classifiers=[
        'License :: Public Domain',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Natural Language :: English',
        'Environment :: Console',
        ],
    zip_safe=False,
    ext_modules=cythonize('beeclust/*.pyx', language='c++', language_level=3),
    include_dirs=[numpy.get_include()],
    setup_requires=[
        'Cython',
        'NumPy',
    ],
    install_requires=[
        'NumPy',
    ],
)
