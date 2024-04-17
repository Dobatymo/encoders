import platform
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

machine = platform.machine().lower()
x86 = ("x86_64", "amd64", "i386", "x86", "i686")

if sys.platform.startswith("linux"):
    if machine in x86:
        cflags = ["-std=c++14", "-O2", "-mavx", "-fopenmp", "-ffast-math"]
    else:
        cflags = ["-std=c++14", "-O2", "-fopenmp", "-ffast-math"]
elif sys.platform == "win32":
    if machine in x86:
        cflags = ["/std:c++14", "/O2", "/arch:AVX", "/openmp", "/fp:fast"]
    else:
        cflags = ["/std:c++14", "/O2", "/openmp", "/fp:fast"]
elif sys.platform == "darwin":
    if machine in x86:
        cflags = ["-std=c++14", "-O2", "-mavx", "-ffast-math"]
    else:
        cflags = ["-std=c++14", "-O2", "-ffast-math"]
else:
    cflags = []

cy_extensions = [
    Extension(
        "encoders.cyfuncs",
        ["encoders/cyfuncs.pyx"],
        include_dirs=[np.get_include(), "encoders"],
        extra_compile_args=cflags,
        language="c++",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

compiler_directives = {
    "binding": False,
    "boundscheck": False,
    "wraparound": False,
    "annotation_typing": True,
    "warn.undeclared": True,
    "warn.unused": True,
    "warn.unused_arg": True,
    "warn.unused_result": True,
}

if __name__ == "__main__":
    setup(
        ext_modules=cythonize(cy_extensions, language_level=3, compiler_directives=compiler_directives),
    )
