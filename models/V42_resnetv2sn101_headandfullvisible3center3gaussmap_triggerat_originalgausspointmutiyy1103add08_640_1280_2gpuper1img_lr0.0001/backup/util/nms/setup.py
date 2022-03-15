from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    name="gpu_nms",  # 生成的扩展模块的名字
    sources=["gpu_nms.pyx"], # 源文件，可以是多个
)
setup(ext_modules=cythonize(ext, language_level=3))  # 指定Python3