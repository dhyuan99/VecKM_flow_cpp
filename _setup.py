from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "VecKM_flow",
        sources=["_binding.cpp"],
        include_dirs=[pybind11.get_include(), "."],
        libraries=["SliceNormalFlowEstimator"],
        library_dirs=["."],
        extra_compile_args=["-O3"],
        language="c++"
    )
]

setup(
    name="VecKM_flow",
    ext_modules=ext_modules,
)
