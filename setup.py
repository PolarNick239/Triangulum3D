from setuptools import setup, find_packages

setup(
    name='Triangulum3D',
    version='0.0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    setup_requires=[
        'setuptools >= 18.0'
    ],
    install_requires=[
        'numpy>=1.9.1',
        'PyOpenGL>=3.1.0',
        'PyYAML>=3.11',
        'pyopencl>=2015.2.3',
    ],
    tests_require=[
        'testfixtures>=4.1.2',
        'nose>=1.3.4'
    ],
    scripts=[
        "monocerosd.py",
    ],
    test_suite='nose.collector',
)
