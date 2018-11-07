from setuptools import setup, find_packages

setup(
    name='beeclust',
    version='0.1',
    description='BeeClust simulation',
    author='Martin Pitak',
    author_email='pitakma1@fit.cvut.cz',
    keywords='numpy,console,simulation',
    license='Public Domain',
    url='https://github.com/EnyMan/beeclust',
    packages=find_packages(),
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
    install_requires=['numpy'],
)
