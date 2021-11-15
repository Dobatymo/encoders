# distutils: language=c++

from cpython.ref cimport PyObject


cdef extern from "pyunicode.hpp":
	cdef cppclass PyUnicodeSmartPtr:

		PyUnicodeSmartPtr(PyObject *) except +
		PyObject *get() const
