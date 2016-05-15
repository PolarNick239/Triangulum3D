from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name='Triangulum3D',
    version='0.0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    setup_requires=[
        'setuptools>=18.0',
        'Cython>=0.24',
    ],
    install_requires=[
        'numpy>=1.9.1',
        'PyOpenGL>=3.1.0',
        'PyYAML>=3.11',
        'pyopencl>=2015.2.3',
        'Pillow>=3.0.0',
        'Cython>=0.24',
        'b2ac>=0.2.1',
    ],

    # TODO: actualize, when pip will support alternative (direct dependencies, see https://github.com/pypa/pip/issues/2023 )
    dependency_links=['https://github.com/PolarNick239/b2ac/tarball/master#egg=b2ac-0.2.1'],

    ext_modules=cythonize("**/*.pyx"),
    tests_require=[
        'testfixtures>=4.1.2',
        'nose>=1.3.4'
    ],
    test_suite='nose.collector',
)
