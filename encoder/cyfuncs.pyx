from cython.parallel import prange
from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from libcpp.utility cimport pair
from cython cimport view
import numpy as np

from cpython.object cimport PyTypeObject
from cpython.unicode cimport PyUnicode_DATA, PyUnicode_GET_LENGTH, PyUnicode_KIND, PyUnicode_FromKindAndData, PyUnicode_READY
from cpython.ref cimport PyObject

from .pyunicode cimport PyUnicode

cdef class BytesLabelEncoder:

	cdef map[string, size_t] _classes

	def __cinit__(self):
		self._classes = map[string, size_t]()

	def __init__(self):
		pass

	def partial_fit(self, seq):
		cdef string item

		for item in seq:
			self._classes.insert(pair[string, size_t](item, self._classes.size()))

	def transform(self, seq):
		cdef string item
		cdef int i

		cdef size_t[::1] out = view.array(shape=(len(seq), ), itemsize=sizeof(size_t), format="Q")

		for i, item in enumerate(seq):
			out[i] = self._classes[item]

		return np.asarray(out)

	@property
	def classes(self):
		return self._classes

cdef PyUnicode _py_str_to_cpp_string(object o):
	cdef int kind = PyUnicode_KIND(o)
	return PyUnicode(kind, string(<char *>PyUnicode_DATA(o), PyUnicode_GET_LENGTH(o) * kind))

cdef object _cpp_string_to_py_str(const PyUnicode &s):
	return PyUnicode_FromKindAndData(s.kind, <void *>s.unicode.data(), s.unicode.size() // s.kind)

cdef class StringLabelEncoder:

	cdef unordered_map[PyUnicode, size_t] _classes

	def __cinit__(self):
		self._classes = unordered_map[PyUnicode, size_t]()

	def __init__(self):
		pass

	def partial_fit(self, seq):
		cdef object item
		cdef PyUnicode tmp

		for item in seq:
			if not isinstance(item, str):
				raise TypeError(f"expected bytes, {type(item)} found")

			tmp = _py_str_to_cpp_string(item)
			self._classes.insert(pair[PyUnicode, size_t](tmp, self._classes.size()))

	def transform(self, seq):
		cdef object item
		cdef int i
		cdef PyUnicode key

		cdef size_t[::1] out = view.array(shape=(len(seq), ), itemsize=sizeof(size_t), format="Q")

		for i, item in enumerate(seq):
			if not isinstance(item, str):
				raise TypeError(f"expected bytes, {type(item)} found")

			key = _py_str_to_cpp_string(item)
			out[i] = self._classes[key]

		return np.asarray(out)

	@property
	def classes(self):
		cdef dict d = dict()
		cdef pair[PyUnicode, size_t] item
		cdef object key

		for item in self._classes:
			key = _cpp_string_to_py_str(item.first)
			d[key] = item.second

		return d
