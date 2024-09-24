from setuptools import setup
from Cython.Build import cythonize
setup(
    ext_modules = cythonize("/home/annone/ai/backend/stream/test_main.pyx")
)
#  python setup.py build_ext --inplace