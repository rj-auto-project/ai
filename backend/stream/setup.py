from setuptools import setup
from Cython.Build import cythonize
setup(
    ext_modules = cythonize("/home/annone/ai-camera/backend/stream/module.pyx")
)
#  python setup.py build_ext --inplace