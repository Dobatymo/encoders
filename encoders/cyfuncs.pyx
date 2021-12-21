from cython.parallel import prange

from cython cimport view
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair

import numpy as np

from cpython.ref cimport PyObject

from .pyunicode cimport PyUnicodeSmartPtr


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

	def fit(self, seq):
		self.partial_fit(seq)

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

cdef class StringLabelEncoder:

	cdef unordered_map[PyUnicodeSmartPtr, size_t] _classes

	def __cinit__(self):
		self._classes = unordered_map[PyUnicodeSmartPtr, size_t]()

	def __init__(self):
		pass

	def partial_fit(self, seq):
		cdef object item

		for item in seq:
			if not isinstance(item, str):
				raise TypeError(f"expected bytes, {type(item)} found")

			self._classes.insert(pair[PyUnicodeSmartPtr, size_t](PyUnicodeSmartPtr(<PyObject *>item), self._classes.size()))

	def fit(self, seq):
		self.partial_fit(seq)

	def transform(self, seq):
		cdef object item
		cdef int i

		cdef size_t[::1] out = view.array(shape=(len(seq), ), itemsize=sizeof(size_t), format="Q")

		for i, item in enumerate(seq):
			if not isinstance(item, str):
				raise TypeError(f"expected bytes, {type(item)} found")

			out[i] = self._classes[PyUnicodeSmartPtr(<PyObject *>item)]

		return np.asarray(out)

	@property
	def classes(self):
		cdef dict d = dict()
		cdef pair[PyUnicodeSmartPtr, size_t] item

		for item in self._classes:
			d[<object>item.first.get()] = item.second

		return d
