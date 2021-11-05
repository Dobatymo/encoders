from libcpp.string cimport string

cdef extern from "pyunicode.hpp":
	cdef cppclass PyUnicode:
		int kind
		string unicode

		PyUnicode()
		PyUnicode(int, string)
