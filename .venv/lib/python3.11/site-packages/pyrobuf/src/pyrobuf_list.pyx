from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.stdint cimport *
from cpython.ref cimport PyObject
from cpython.object cimport Py_EQ, Py_NE


cdef class TypedList(list):

    def __init__(self, type list_type, object listener=None):
        self._list_type = list_type
        self._listener = listener

    property list_type:
        def __get__(self):
            return self._list_type

    def __setitem__(self, i, x):
        try:
            assert isinstance(x, self._list_type)
        except AssertionError:
            raise Exception("type mismatch")

        super(TypedList, self).__setitem__(i, x)
        if self._listener is not None:
            self._listener()



    def add(self, **kwargs):
        elt = self._list_type(**kwargs)
        super(TypedList, self).append(elt)
        if self._listener is not None:
            self._listener()
        return elt

    def append(self, x):
        try:
            assert isinstance(x, self._list_type)
        except AssertionError:
            raise Exception("type mismatch")

        super(TypedList, self).append(x)
        if self._listener is not None:
            self._listener()

    def extend(self, x):
        for i in x:
            elt = self._list_type()
            elt.MergeFrom(i)
            super(TypedList, self).append(elt)

        if self._listener is not None:
            self._listener()

    def insert(self, i, x):
        try:
            assert isinstance(x, self._list_type)
        except AssertionError:
            raise Exception("type mismatch")

        super(TypedList, self).insert(i, x)
        if self._listener is not None:
            self._listener()


cdef class BytesList(list):

    def __init__(self, object listener=None):
        self._listener = listener

    def __setitem__(self, i, x):
        cdef list values

        if isinstance(i, int):
            if not isinstance(x, bytes):
                raise TypeError("%r has type %s, but expected one of: (%s,)" % (x, type(x), bytes))

            list.__setitem__(self, i, x)
        elif isinstance(i, slice):
            values = list()

            for val in x:
                if isinstance(val, bytes):
                    values.append(val)
                else:
                    raise TypeError("%r has type %s, but expected one of: (%s,)" % (val, type(val), bytes))

            list.__setitem__(self, i, values)
        else:
            raise TypeError("list indices must be integers, not %s" % type(i).__name__)

        if self._listener is not None:
            self._listener()



    def append(self, x):
        if not isinstance(x, bytes):
            raise TypeError("%r has type %s, but expected one of: (%s,)" % (x, type(x), bytes))

        list.append(self, x)
        if self._listener is not None:
            self._listener()

    def extend(self, x):
        for val in x:
            if not isinstance(val, bytes):
                raise TypeError("%r has type %s, but expected one of: (%s,)" % (val, type(val), bytes))
            list.append(self, val)

        if self._listener is not None:
            self._listener()

    def insert(self, i, x):

        if not isinstance(x, bytes):
            raise TypeError("%r has type %s, but expected one of: (%s,)" % (x, type(x), bytes))

        list.insert(self, i, x)
        if self._listener is not None:
            self._listener()


cdef class StringList(list):

    def __init__(self, object listener=None):
        self._listener = listener

    def __setitem__(self, i, x):
        cdef list values
        cdef str value

        if isinstance(i, int):
            if isinstance(x, str):
                value = x
            elif isinstance(x, bytes):
                value = x.decode('utf-8')
            else:
                raise TypeError("%r has type %s, but expected one of: (%s, %s)" % (x, type(x), bytes, str))

            list.__setitem__(self, i, value)
        elif isinstance(i, slice):
            values = list()

            for val in x:
                if isinstance(val, str):
                    values.append(val)
                elif isinstance(val, bytes):
                    values.append(val.decode('utf-8'))
                else:
                    raise TypeError("%r has type %s, but expected one of: (%s, %s)" % (val, type(val), bytes, str))

            list.__setitem__(self, i, values)
        else:
            raise TypeError("list indices must be integers, not %s" % type(i).__name__)

        if self._listener is not None:
            self._listener()



    def append(self, x):
        cdef str value
        if isinstance(x, str):
            value = x
        elif isinstance(x, bytes):
            value = x.decode('utf-8')
        else:
            raise TypeError("%r has type %s, but expected one of: (%s, %s)" % (x, type(x), bytes, str))

        list.append(self, value)
        if self._listener is not None:
            self._listener()

    def extend(self, x):
        cdef str value

        for val in x:
            if isinstance(val, bytes):
                value = val.decode('utf-8')
            elif isinstance(val, str):
                value = val
            else:
                raise TypeError("%r has type %s, but expected one of: (%s, %s)" % (val, type(val), bytes, str))
            list.append(self, value)

        if self._listener is not None:
            self._listener()

    def insert(self, i, x):
        cdef str value
        if isinstance(x, bytes):
            value = x.decode('utf-8')
        elif isinstance(x, str):
            value = x
        else:
            raise TypeError("%r has type %s, but expected one of: (%s, %s)" % (x, type(x), bytes, str))

        list.insert(self, i, value)
        if self._listener is not None:
            self._listener()






cdef class DoubleList:

    def __cinit__(self, size_t size=16, object listener=None):
        self._data = <double *>PyMem_Malloc(size * sizeof(double))
        if not self._data:
            raise MemoryError()

        self._n_items = 0
        self._size = size
        self._listener = listener

    def __dealloc__(self):
        PyMem_Free(self._data)

    def __contains__(self, double x):
        cdef size_t i
        for i in range(self._n_items):
            if self._data[i] == x:
                return True

        return False

    def __delitem__(self, int i):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items:
            raise IndexError("list index out of range")

        cdef size_t j
        for j in range(i, self._n_items):
            self._data[j] = self._data[j + 1]

        self._n_items -= 1
        if self._listener is not None:
            self._listener()

    def __getitem__(self, int i):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items or i < 0:
            raise IndexError("list index out of range")

        return self._data[i]

    def __iter__(self):
        cdef size_t idx = 0
        while idx < self._n_items:
            yield self._data[idx]
            idx += 1

    def __len__(self):
        return self._n_items

    def __repr__(self):
        return repr(list(self))

    def __setitem__(self, int i, double x):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items:
            raise IndexError("list index out of range")

        self._data[i] = x
        if self._listener is not None:
            self._listener()

    def __str__(self):
        return str(list(self))

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        
        cdef Py_ssize_t itemsize = sizeof(double)
        self.shape[0] = self._n_items
        self.strides[0] = 1

        buffer.buf = <char *>(self._data)
        buffer.format = 'd'
        buffer.internal = NULL                  # see References
        buffer.itemsize = itemsize
        buffer.len = self._n_items
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 1
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL                # for pointer arrays only

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __richcmp__(self, other, op):
        if op == Py_EQ:
            return self._equal(other)
        elif op == Py_NE:
            return not self._equal(other)
        else:
            raise TypeError('Operation not supported for lists')

    def __neq__(self, other):
        return not self == other

    cpdef _equal(self, other):
        try:
            if len(self) != len(other):
                return False
        except TypeError:
            return False

        for x, y in zip(self, other):
            if x != y:
                return False

        return True

    cpdef append(self, double x):
        self._append(x)
        if self._listener is not None:
            self._listener()

    cpdef extend(self, x):
        for i in x:
            self._append(i)

        if self._listener is not None:
            self._listener()

    cpdef insert(self, int i, double x):
        cdef double *mem
        cdef size_t j

        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i > self._n_items:
            raise IndexError("list index out of range")

        if self._n_items == self._size:
            mem = <double *>PyMem_Realloc(self._data, 2 * self._size * sizeof(double))
            if not mem:
                raise MemoryError()

            self._data = mem
            self._size *= 2

        for j in range(self._n_items, <size_t>i, -1):
            self._data[j] = self._data[j - 1]

        self._data[i] = x
        self._n_items += 1
        if self._listener is not None:
            self._listener()

    cpdef pop(self):
        if self._n_items == 0:
            raise IndexError("pop from empty list")

        self._n_items -= 1
        if self._listener is not None:
            self._listener()
        return self._data[self._n_items]

    cpdef remove(self, double x):
        cdef size_t i
        cdef size_t j
        cdef bint found = 0

        for i in range(self._n_items):
            if self._data[i] == x:
                found = 1
                break

        if found == 0:
            raise ValueError("x not in list")

        for j in range(i, self._n_items):
            self._data[j] = self._data[j + 1]

        self._n_items -= 1
        if self._listener is not None:
            self._listener()

    cdef void _append(self, double x):
        cdef double *mem

        if self._n_items == self._size:
            mem = <double *>PyMem_Realloc(self._data, 2 * self._size * sizeof(double))
            if not mem:
                raise MemoryError()

            self._data = mem
            self._size *= 2

        self._data[self._n_items] = x
        self._n_items += 1



cdef class FloatList:

    def __cinit__(self, size_t size=16, object listener=None):
        self._data = <float *>PyMem_Malloc(size * sizeof(float))
        if not self._data:
            raise MemoryError()

        self._n_items = 0
        self._size = size
        self._listener = listener

    def __dealloc__(self):
        PyMem_Free(self._data)

    def __contains__(self, float x):
        cdef size_t i
        for i in range(self._n_items):
            if self._data[i] == x:
                return True

        return False

    def __delitem__(self, int i):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items:
            raise IndexError("list index out of range")

        cdef size_t j
        for j in range(i, self._n_items):
            self._data[j] = self._data[j + 1]

        self._n_items -= 1
        if self._listener is not None:
            self._listener()

    def __getitem__(self, int i):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items or i < 0:
            raise IndexError("list index out of range")

        return self._data[i]

    def __iter__(self):
        cdef size_t idx = 0
        while idx < self._n_items:
            yield self._data[idx]
            idx += 1

    def __len__(self):
        return self._n_items

    def __repr__(self):
        return repr(list(self))

    def __setitem__(self, int i, float x):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items:
            raise IndexError("list index out of range")

        self._data[i] = x
        if self._listener is not None:
            self._listener()

    def __str__(self):
        return str(list(self))

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        
        cdef Py_ssize_t itemsize = sizeof(float)
        self.shape[0] = self._n_items
        self.strides[0] = 1

        buffer.buf = <char *>(self._data)
        buffer.format = 'f'
        buffer.internal = NULL                  # see References
        buffer.itemsize = itemsize
        buffer.len = self._n_items
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 1
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL                # for pointer arrays only

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __richcmp__(self, other, op):
        if op == Py_EQ:
            return self._equal(other)
        elif op == Py_NE:
            return not self._equal(other)
        else:
            raise TypeError('Operation not supported for lists')

    def __neq__(self, other):
        return not self == other

    cpdef _equal(self, other):
        try:
            if len(self) != len(other):
                return False
        except TypeError:
            return False

        for x, y in zip(self, other):
            if x != y:
                return False

        return True

    cpdef append(self, float x):
        self._append(x)
        if self._listener is not None:
            self._listener()

    cpdef extend(self, x):
        for i in x:
            self._append(i)

        if self._listener is not None:
            self._listener()

    cpdef insert(self, int i, float x):
        cdef float *mem
        cdef size_t j

        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i > self._n_items:
            raise IndexError("list index out of range")

        if self._n_items == self._size:
            mem = <float *>PyMem_Realloc(self._data, 2 * self._size * sizeof(float))
            if not mem:
                raise MemoryError()

            self._data = mem
            self._size *= 2

        for j in range(self._n_items, <size_t>i, -1):
            self._data[j] = self._data[j - 1]

        self._data[i] = x
        self._n_items += 1
        if self._listener is not None:
            self._listener()

    cpdef pop(self):
        if self._n_items == 0:
            raise IndexError("pop from empty list")

        self._n_items -= 1
        if self._listener is not None:
            self._listener()
        return self._data[self._n_items]

    cpdef remove(self, float x):
        cdef size_t i
        cdef size_t j
        cdef bint found = 0

        for i in range(self._n_items):
            if self._data[i] == x:
                found = 1
                break

        if found == 0:
            raise ValueError("x not in list")

        for j in range(i, self._n_items):
            self._data[j] = self._data[j + 1]

        self._n_items -= 1
        if self._listener is not None:
            self._listener()

    cdef void _append(self, float x):
        cdef float *mem

        if self._n_items == self._size:
            mem = <float *>PyMem_Realloc(self._data, 2 * self._size * sizeof(float))
            if not mem:
                raise MemoryError()

            self._data = mem
            self._size *= 2

        self._data[self._n_items] = x
        self._n_items += 1



cdef class IntList:

    def __cinit__(self, size_t size=16, object listener=None):
        self._data = <int *>PyMem_Malloc(size * sizeof(int))
        if not self._data:
            raise MemoryError()

        self._n_items = 0
        self._size = size
        self._listener = listener

    def __dealloc__(self):
        PyMem_Free(self._data)

    def __contains__(self, int x):
        cdef size_t i
        for i in range(self._n_items):
            if self._data[i] == x:
                return True

        return False

    def __delitem__(self, int i):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items:
            raise IndexError("list index out of range")

        cdef size_t j
        for j in range(i, self._n_items):
            self._data[j] = self._data[j + 1]

        self._n_items -= 1
        if self._listener is not None:
            self._listener()

    def __getitem__(self, int i):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items or i < 0:
            raise IndexError("list index out of range")

        return self._data[i]

    def __iter__(self):
        cdef size_t idx = 0
        while idx < self._n_items:
            yield self._data[idx]
            idx += 1

    def __len__(self):
        return self._n_items

    def __repr__(self):
        return repr(list(self))

    def __setitem__(self, int i, int x):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items:
            raise IndexError("list index out of range")

        self._data[i] = x
        if self._listener is not None:
            self._listener()

    def __str__(self):
        return str(list(self))

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        
        cdef Py_ssize_t itemsize = sizeof(int)
        self.shape[0] = self._n_items
        self.strides[0] = 1

        buffer.buf = <char *>(self._data)
        buffer.format = 'i'
        buffer.internal = NULL                  # see References
        buffer.itemsize = itemsize
        buffer.len = self._n_items
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 1
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL                # for pointer arrays only

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __richcmp__(self, other, op):
        if op == Py_EQ:
            return self._equal(other)
        elif op == Py_NE:
            return not self._equal(other)
        else:
            raise TypeError('Operation not supported for lists')

    def __neq__(self, other):
        return not self == other

    cpdef _equal(self, other):
        try:
            if len(self) != len(other):
                return False
        except TypeError:
            return False

        for x, y in zip(self, other):
            if x != y:
                return False

        return True

    cpdef append(self, int x):
        self._append(x)
        if self._listener is not None:
            self._listener()

    cpdef extend(self, x):
        for i in x:
            self._append(i)

        if self._listener is not None:
            self._listener()

    cpdef insert(self, int i, int x):
        cdef int *mem
        cdef size_t j

        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i > self._n_items:
            raise IndexError("list index out of range")

        if self._n_items == self._size:
            mem = <int *>PyMem_Realloc(self._data, 2 * self._size * sizeof(int))
            if not mem:
                raise MemoryError()

            self._data = mem
            self._size *= 2

        for j in range(self._n_items, <size_t>i, -1):
            self._data[j] = self._data[j - 1]

        self._data[i] = x
        self._n_items += 1
        if self._listener is not None:
            self._listener()

    cpdef pop(self):
        if self._n_items == 0:
            raise IndexError("pop from empty list")

        self._n_items -= 1
        if self._listener is not None:
            self._listener()
        return self._data[self._n_items]

    cpdef remove(self, int x):
        cdef size_t i
        cdef size_t j
        cdef bint found = 0

        for i in range(self._n_items):
            if self._data[i] == x:
                found = 1
                break

        if found == 0:
            raise ValueError("x not in list")

        for j in range(i, self._n_items):
            self._data[j] = self._data[j + 1]

        self._n_items -= 1
        if self._listener is not None:
            self._listener()

    cdef void _append(self, int x):
        cdef int *mem

        if self._n_items == self._size:
            mem = <int *>PyMem_Realloc(self._data, 2 * self._size * sizeof(int))
            if not mem:
                raise MemoryError()

            self._data = mem
            self._size *= 2

        self._data[self._n_items] = x
        self._n_items += 1



cdef class Int32List:

    def __cinit__(self, size_t size=16, object listener=None):
        self._data = <int32_t *>PyMem_Malloc(size * sizeof(int32_t))
        if not self._data:
            raise MemoryError()

        self._n_items = 0
        self._size = size
        self._listener = listener

    def __dealloc__(self):
        PyMem_Free(self._data)

    def __contains__(self, int32_t x):
        cdef size_t i
        for i in range(self._n_items):
            if self._data[i] == x:
                return True

        return False

    def __delitem__(self, int i):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items:
            raise IndexError("list index out of range")

        cdef size_t j
        for j in range(i, self._n_items):
            self._data[j] = self._data[j + 1]

        self._n_items -= 1
        if self._listener is not None:
            self._listener()

    def __getitem__(self, int i):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items or i < 0:
            raise IndexError("list index out of range")

        return self._data[i]

    def __iter__(self):
        cdef size_t idx = 0
        while idx < self._n_items:
            yield self._data[idx]
            idx += 1

    def __len__(self):
        return self._n_items

    def __repr__(self):
        return repr(list(self))

    def __setitem__(self, int i, int32_t x):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items:
            raise IndexError("list index out of range")

        self._data[i] = x
        if self._listener is not None:
            self._listener()

    def __str__(self):
        return str(list(self))

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        
        cdef Py_ssize_t itemsize = sizeof(int32_t)
        self.shape[0] = self._n_items
        self.strides[0] = 1

        buffer.buf = <char *>(self._data)
        buffer.format = 'i'
        buffer.internal = NULL                  # see References
        buffer.itemsize = itemsize
        buffer.len = self._n_items
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 1
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL                # for pointer arrays only

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __richcmp__(self, other, op):
        if op == Py_EQ:
            return self._equal(other)
        elif op == Py_NE:
            return not self._equal(other)
        else:
            raise TypeError('Operation not supported for lists')

    def __neq__(self, other):
        return not self == other

    cpdef _equal(self, other):
        try:
            if len(self) != len(other):
                return False
        except TypeError:
            return False

        for x, y in zip(self, other):
            if x != y:
                return False

        return True

    cpdef append(self, int32_t x):
        self._append(x)
        if self._listener is not None:
            self._listener()

    cpdef extend(self, x):
        for i in x:
            self._append(i)

        if self._listener is not None:
            self._listener()

    cpdef insert(self, int i, int32_t x):
        cdef int32_t *mem
        cdef size_t j

        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i > self._n_items:
            raise IndexError("list index out of range")

        if self._n_items == self._size:
            mem = <int32_t *>PyMem_Realloc(self._data, 2 * self._size * sizeof(int32_t))
            if not mem:
                raise MemoryError()

            self._data = mem
            self._size *= 2

        for j in range(self._n_items, <size_t>i, -1):
            self._data[j] = self._data[j - 1]

        self._data[i] = x
        self._n_items += 1
        if self._listener is not None:
            self._listener()

    cpdef pop(self):
        if self._n_items == 0:
            raise IndexError("pop from empty list")

        self._n_items -= 1
        if self._listener is not None:
            self._listener()
        return self._data[self._n_items]

    cpdef remove(self, int32_t x):
        cdef size_t i
        cdef size_t j
        cdef bint found = 0

        for i in range(self._n_items):
            if self._data[i] == x:
                found = 1
                break

        if found == 0:
            raise ValueError("x not in list")

        for j in range(i, self._n_items):
            self._data[j] = self._data[j + 1]

        self._n_items -= 1
        if self._listener is not None:
            self._listener()

    cdef void _append(self, int32_t x):
        cdef int32_t *mem

        if self._n_items == self._size:
            mem = <int32_t *>PyMem_Realloc(self._data, 2 * self._size * sizeof(int32_t))
            if not mem:
                raise MemoryError()

            self._data = mem
            self._size *= 2

        self._data[self._n_items] = x
        self._n_items += 1



cdef class Uint32List:

    def __cinit__(self, size_t size=16, object listener=None):
        self._data = <uint32_t *>PyMem_Malloc(size * sizeof(uint32_t))
        if not self._data:
            raise MemoryError()

        self._n_items = 0
        self._size = size
        self._listener = listener

    def __dealloc__(self):
        PyMem_Free(self._data)

    def __contains__(self, uint32_t x):
        cdef size_t i
        for i in range(self._n_items):
            if self._data[i] == x:
                return True

        return False

    def __delitem__(self, int i):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items:
            raise IndexError("list index out of range")

        cdef size_t j
        for j in range(i, self._n_items):
            self._data[j] = self._data[j + 1]

        self._n_items -= 1
        if self._listener is not None:
            self._listener()

    def __getitem__(self, int i):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items or i < 0:
            raise IndexError("list index out of range")

        return self._data[i]

    def __iter__(self):
        cdef size_t idx = 0
        while idx < self._n_items:
            yield self._data[idx]
            idx += 1

    def __len__(self):
        return self._n_items

    def __repr__(self):
        return repr(list(self))

    def __setitem__(self, int i, uint32_t x):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items:
            raise IndexError("list index out of range")

        self._data[i] = x
        if self._listener is not None:
            self._listener()

    def __str__(self):
        return str(list(self))

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        
        cdef Py_ssize_t itemsize = sizeof(uint32_t)
        self.shape[0] = self._n_items
        self.strides[0] = 1

        buffer.buf = <char *>(self._data)
        buffer.format = 'I'
        buffer.internal = NULL                  # see References
        buffer.itemsize = itemsize
        buffer.len = self._n_items
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 1
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL                # for pointer arrays only

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __richcmp__(self, other, op):
        if op == Py_EQ:
            return self._equal(other)
        elif op == Py_NE:
            return not self._equal(other)
        else:
            raise TypeError('Operation not supported for lists')

    def __neq__(self, other):
        return not self == other

    cpdef _equal(self, other):
        try:
            if len(self) != len(other):
                return False
        except TypeError:
            return False

        for x, y in zip(self, other):
            if x != y:
                return False

        return True

    cpdef append(self, uint32_t x):
        self._append(x)
        if self._listener is not None:
            self._listener()

    cpdef extend(self, x):
        for i in x:
            self._append(i)

        if self._listener is not None:
            self._listener()

    cpdef insert(self, int i, uint32_t x):
        cdef uint32_t *mem
        cdef size_t j

        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i > self._n_items:
            raise IndexError("list index out of range")

        if self._n_items == self._size:
            mem = <uint32_t *>PyMem_Realloc(self._data, 2 * self._size * sizeof(uint32_t))
            if not mem:
                raise MemoryError()

            self._data = mem
            self._size *= 2

        for j in range(self._n_items, <size_t>i, -1):
            self._data[j] = self._data[j - 1]

        self._data[i] = x
        self._n_items += 1
        if self._listener is not None:
            self._listener()

    cpdef pop(self):
        if self._n_items == 0:
            raise IndexError("pop from empty list")

        self._n_items -= 1
        if self._listener is not None:
            self._listener()
        return self._data[self._n_items]

    cpdef remove(self, uint32_t x):
        cdef size_t i
        cdef size_t j
        cdef bint found = 0

        for i in range(self._n_items):
            if self._data[i] == x:
                found = 1
                break

        if found == 0:
            raise ValueError("x not in list")

        for j in range(i, self._n_items):
            self._data[j] = self._data[j + 1]

        self._n_items -= 1
        if self._listener is not None:
            self._listener()

    cdef void _append(self, uint32_t x):
        cdef uint32_t *mem

        if self._n_items == self._size:
            mem = <uint32_t *>PyMem_Realloc(self._data, 2 * self._size * sizeof(uint32_t))
            if not mem:
                raise MemoryError()

            self._data = mem
            self._size *= 2

        self._data[self._n_items] = x
        self._n_items += 1



cdef class Int64List:

    def __cinit__(self, size_t size=16, object listener=None):
        self._data = <int64_t *>PyMem_Malloc(size * sizeof(int64_t))
        if not self._data:
            raise MemoryError()

        self._n_items = 0
        self._size = size
        self._listener = listener

    def __dealloc__(self):
        PyMem_Free(self._data)

    def __contains__(self, int64_t x):
        cdef size_t i
        for i in range(self._n_items):
            if self._data[i] == x:
                return True

        return False

    def __delitem__(self, int i):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items:
            raise IndexError("list index out of range")

        cdef size_t j
        for j in range(i, self._n_items):
            self._data[j] = self._data[j + 1]

        self._n_items -= 1
        if self._listener is not None:
            self._listener()

    def __getitem__(self, int i):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items or i < 0:
            raise IndexError("list index out of range")

        return self._data[i]

    def __iter__(self):
        cdef size_t idx = 0
        while idx < self._n_items:
            yield self._data[idx]
            idx += 1

    def __len__(self):
        return self._n_items

    def __repr__(self):
        return repr(list(self))

    def __setitem__(self, int i, int64_t x):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items:
            raise IndexError("list index out of range")

        self._data[i] = x
        if self._listener is not None:
            self._listener()

    def __str__(self):
        return str(list(self))

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        
        cdef Py_ssize_t itemsize = sizeof(int64_t)
        self.shape[0] = self._n_items
        self.strides[0] = 1

        buffer.buf = <char *>(self._data)
        buffer.format = 'L'
        buffer.internal = NULL                  # see References
        buffer.itemsize = itemsize
        buffer.len = self._n_items
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 1
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL                # for pointer arrays only

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __richcmp__(self, other, op):
        if op == Py_EQ:
            return self._equal(other)
        elif op == Py_NE:
            return not self._equal(other)
        else:
            raise TypeError('Operation not supported for lists')

    def __neq__(self, other):
        return not self == other

    cpdef _equal(self, other):
        try:
            if len(self) != len(other):
                return False
        except TypeError:
            return False

        for x, y in zip(self, other):
            if x != y:
                return False

        return True

    cpdef append(self, int64_t x):
        self._append(x)
        if self._listener is not None:
            self._listener()

    cpdef extend(self, x):
        for i in x:
            self._append(i)

        if self._listener is not None:
            self._listener()

    cpdef insert(self, int i, int64_t x):
        cdef int64_t *mem
        cdef size_t j

        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i > self._n_items:
            raise IndexError("list index out of range")

        if self._n_items == self._size:
            mem = <int64_t *>PyMem_Realloc(self._data, 2 * self._size * sizeof(int64_t))
            if not mem:
                raise MemoryError()

            self._data = mem
            self._size *= 2

        for j in range(self._n_items, <size_t>i, -1):
            self._data[j] = self._data[j - 1]

        self._data[i] = x
        self._n_items += 1
        if self._listener is not None:
            self._listener()

    cpdef pop(self):
        if self._n_items == 0:
            raise IndexError("pop from empty list")

        self._n_items -= 1
        if self._listener is not None:
            self._listener()
        return self._data[self._n_items]

    cpdef remove(self, int64_t x):
        cdef size_t i
        cdef size_t j
        cdef bint found = 0

        for i in range(self._n_items):
            if self._data[i] == x:
                found = 1
                break

        if found == 0:
            raise ValueError("x not in list")

        for j in range(i, self._n_items):
            self._data[j] = self._data[j + 1]

        self._n_items -= 1
        if self._listener is not None:
            self._listener()

    cdef void _append(self, int64_t x):
        cdef int64_t *mem

        if self._n_items == self._size:
            mem = <int64_t *>PyMem_Realloc(self._data, 2 * self._size * sizeof(int64_t))
            if not mem:
                raise MemoryError()

            self._data = mem
            self._size *= 2

        self._data[self._n_items] = x
        self._n_items += 1



cdef class Uint64List:

    def __cinit__(self, size_t size=16, object listener=None):
        self._data = <uint64_t *>PyMem_Malloc(size * sizeof(uint64_t))
        if not self._data:
            raise MemoryError()

        self._n_items = 0
        self._size = size
        self._listener = listener

    def __dealloc__(self):
        PyMem_Free(self._data)

    def __contains__(self, uint64_t x):
        cdef size_t i
        for i in range(self._n_items):
            if self._data[i] == x:
                return True

        return False

    def __delitem__(self, int i):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items:
            raise IndexError("list index out of range")

        cdef size_t j
        for j in range(i, self._n_items):
            self._data[j] = self._data[j + 1]

        self._n_items -= 1
        if self._listener is not None:
            self._listener()

    def __getitem__(self, int i):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items or i < 0:
            raise IndexError("list index out of range")

        return self._data[i]

    def __iter__(self):
        cdef size_t idx = 0
        while idx < self._n_items:
            yield self._data[idx]
            idx += 1

    def __len__(self):
        return self._n_items

    def __repr__(self):
        return repr(list(self))

    def __setitem__(self, int i, uint64_t x):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items:
            raise IndexError("list index out of range")

        self._data[i] = x
        if self._listener is not None:
            self._listener()

    def __str__(self):
        return str(list(self))

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        
        cdef Py_ssize_t itemsize = sizeof(uint64_t)
        self.shape[0] = self._n_items
        self.strides[0] = 1

        buffer.buf = <char *>(self._data)
        buffer.format = 'K'
        buffer.internal = NULL                  # see References
        buffer.itemsize = itemsize
        buffer.len = self._n_items
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 1
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL                # for pointer arrays only

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __richcmp__(self, other, op):
        if op == Py_EQ:
            return self._equal(other)
        elif op == Py_NE:
            return not self._equal(other)
        else:
            raise TypeError('Operation not supported for lists')

    def __neq__(self, other):
        return not self == other

    cpdef _equal(self, other):
        try:
            if len(self) != len(other):
                return False
        except TypeError:
            return False

        for x, y in zip(self, other):
            if x != y:
                return False

        return True

    cpdef append(self, uint64_t x):
        self._append(x)
        if self._listener is not None:
            self._listener()

    cpdef extend(self, x):
        for i in x:
            self._append(i)

        if self._listener is not None:
            self._listener()

    cpdef insert(self, int i, uint64_t x):
        cdef uint64_t *mem
        cdef size_t j

        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i > self._n_items:
            raise IndexError("list index out of range")

        if self._n_items == self._size:
            mem = <uint64_t *>PyMem_Realloc(self._data, 2 * self._size * sizeof(uint64_t))
            if not mem:
                raise MemoryError()

            self._data = mem
            self._size *= 2

        for j in range(self._n_items, <size_t>i, -1):
            self._data[j] = self._data[j - 1]

        self._data[i] = x
        self._n_items += 1
        if self._listener is not None:
            self._listener()

    cpdef pop(self):
        if self._n_items == 0:
            raise IndexError("pop from empty list")

        self._n_items -= 1
        if self._listener is not None:
            self._listener()
        return self._data[self._n_items]

    cpdef remove(self, uint64_t x):
        cdef size_t i
        cdef size_t j
        cdef bint found = 0

        for i in range(self._n_items):
            if self._data[i] == x:
                found = 1
                break

        if found == 0:
            raise ValueError("x not in list")

        for j in range(i, self._n_items):
            self._data[j] = self._data[j + 1]

        self._n_items -= 1
        if self._listener is not None:
            self._listener()

    cdef void _append(self, uint64_t x):
        cdef uint64_t *mem

        if self._n_items == self._size:
            mem = <uint64_t *>PyMem_Realloc(self._data, 2 * self._size * sizeof(uint64_t))
            if not mem:
                raise MemoryError()

            self._data = mem
            self._size *= 2

        self._data[self._n_items] = x
        self._n_items += 1



cdef class CharList:

    def __cinit__(self, size_t size=16, object listener=None):
        self._data = <char *>PyMem_Malloc(size * sizeof(char))
        if not self._data:
            raise MemoryError()

        self._n_items = 0
        self._size = size
        self._listener = listener

    def __dealloc__(self):
        PyMem_Free(self._data)

    def __contains__(self, char x):
        cdef size_t i
        for i in range(self._n_items):
            if self._data[i] == x:
                return True

        return False

    def __delitem__(self, int i):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items:
            raise IndexError("list index out of range")

        cdef size_t j
        for j in range(i, self._n_items):
            self._data[j] = self._data[j + 1]

        self._n_items -= 1
        if self._listener is not None:
            self._listener()

    def __getitem__(self, int i):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items or i < 0:
            raise IndexError("list index out of range")

        return self._data[i]

    def __iter__(self):
        cdef size_t idx = 0
        while idx < self._n_items:
            yield self._data[idx]
            idx += 1

    def __len__(self):
        return self._n_items

    def __repr__(self):
        return repr(list(self))

    def __setitem__(self, int i, char x):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items:
            raise IndexError("list index out of range")

        self._data[i] = x
        if self._listener is not None:
            self._listener()

    def __str__(self):
        return str(list(self))

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        
        cdef Py_ssize_t itemsize = sizeof(char)
        self.shape[0] = self._n_items
        self.strides[0] = 1

        buffer.buf = <char *>(self._data)
        buffer.format = 'c'
        buffer.internal = NULL                  # see References
        buffer.itemsize = itemsize
        buffer.len = self._n_items
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 1
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL                # for pointer arrays only

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __richcmp__(self, other, op):
        if op == Py_EQ:
            return self._equal(other)
        elif op == Py_NE:
            return not self._equal(other)
        else:
            raise TypeError('Operation not supported for lists')

    def __neq__(self, other):
        return not self == other

    cpdef _equal(self, other):
        try:
            if len(self) != len(other):
                return False
        except TypeError:
            return False

        for x, y in zip(self, other):
            if x != y:
                return False

        return True

    cpdef append(self, char x):
        self._append(x)
        if self._listener is not None:
            self._listener()

    cpdef extend(self, x):
        for i in x:
            self._append(i)

        if self._listener is not None:
            self._listener()

    cpdef insert(self, int i, char x):
        cdef char *mem
        cdef size_t j

        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i > self._n_items:
            raise IndexError("list index out of range")

        if self._n_items == self._size:
            mem = <char *>PyMem_Realloc(self._data, 2 * self._size * sizeof(char))
            if not mem:
                raise MemoryError()

            self._data = mem
            self._size *= 2

        for j in range(self._n_items, <size_t>i, -1):
            self._data[j] = self._data[j - 1]

        self._data[i] = x
        self._n_items += 1
        if self._listener is not None:
            self._listener()

    cpdef pop(self):
        if self._n_items == 0:
            raise IndexError("pop from empty list")

        self._n_items -= 1
        if self._listener is not None:
            self._listener()
        return self._data[self._n_items]

    cpdef remove(self, char x):
        cdef size_t i
        cdef size_t j
        cdef bint found = 0

        for i in range(self._n_items):
            if self._data[i] == x:
                found = 1
                break

        if found == 0:
            raise ValueError("x not in list")

        for j in range(i, self._n_items):
            self._data[j] = self._data[j + 1]

        self._n_items -= 1
        if self._listener is not None:
            self._listener()

    cdef void _append(self, char x):
        cdef char *mem

        if self._n_items == self._size:
            mem = <char *>PyMem_Realloc(self._data, 2 * self._size * sizeof(char))
            if not mem:
                raise MemoryError()

            self._data = mem
            self._size *= 2

        self._data[self._n_items] = x
        self._n_items += 1



cdef class BintList:

    def __cinit__(self, size_t size=16, object listener=None):
        self._data = <bint *>PyMem_Malloc(size * sizeof(bint))
        if not self._data:
            raise MemoryError()

        self._n_items = 0
        self._size = size
        self._listener = listener

    def __dealloc__(self):
        PyMem_Free(self._data)

    def __contains__(self, bint x):
        cdef size_t i
        for i in range(self._n_items):
            if self._data[i] == x:
                return True

        return False

    def __delitem__(self, int i):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items:
            raise IndexError("list index out of range")

        cdef size_t j
        for j in range(i, self._n_items):
            self._data[j] = self._data[j + 1]

        self._n_items -= 1
        if self._listener is not None:
            self._listener()

    def __getitem__(self, int i):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items or i < 0:
            raise IndexError("list index out of range")

        return self._data[i]

    def __iter__(self):
        cdef size_t idx = 0
        while idx < self._n_items:
            yield self._data[idx]
            idx += 1

    def __len__(self):
        return self._n_items

    def __repr__(self):
        return repr(list(self))

    def __setitem__(self, int i, bint x):
        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i >= self._n_items:
            raise IndexError("list index out of range")

        self._data[i] = x
        if self._listener is not None:
            self._listener()

    def __str__(self):
        return str(list(self))

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        
        cdef Py_ssize_t itemsize = sizeof(bint)
        self.shape[0] = self._n_items
        self.strides[0] = 1

        buffer.buf = <char *>(self._data)
        buffer.format = ''
        buffer.internal = NULL                  # see References
        buffer.itemsize = itemsize
        buffer.len = self._n_items
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 1
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL                # for pointer arrays only

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __richcmp__(self, other, op):
        if op == Py_EQ:
            return self._equal(other)
        elif op == Py_NE:
            return not self._equal(other)
        else:
            raise TypeError('Operation not supported for lists')

    def __neq__(self, other):
        return not self == other

    cpdef _equal(self, other):
        try:
            if len(self) != len(other):
                return False
        except TypeError:
            return False

        for x, y in zip(self, other):
            if x != y:
                return False

        return True

    cpdef append(self, bint x):
        self._append(x)
        if self._listener is not None:
            self._listener()

    cpdef extend(self, x):
        for i in x:
            self._append(i)

        if self._listener is not None:
            self._listener()

    cpdef insert(self, int i, bint x):
        cdef bint *mem
        cdef size_t j

        if i < 0:
            i += <int>self._n_items

        if i < 0 or <size_t>i > self._n_items:
            raise IndexError("list index out of range")

        if self._n_items == self._size:
            mem = <bint *>PyMem_Realloc(self._data, 2 * self._size * sizeof(bint))
            if not mem:
                raise MemoryError()

            self._data = mem
            self._size *= 2

        for j in range(self._n_items, <size_t>i, -1):
            self._data[j] = self._data[j - 1]

        self._data[i] = x
        self._n_items += 1
        if self._listener is not None:
            self._listener()

    cpdef pop(self):
        if self._n_items == 0:
            raise IndexError("pop from empty list")

        self._n_items -= 1
        if self._listener is not None:
            self._listener()
        return self._data[self._n_items]

    cpdef remove(self, bint x):
        cdef size_t i
        cdef size_t j
        cdef bint found = 0

        for i in range(self._n_items):
            if self._data[i] == x:
                found = 1
                break

        if found == 0:
            raise ValueError("x not in list")

        for j in range(i, self._n_items):
            self._data[j] = self._data[j + 1]

        self._n_items -= 1
        if self._listener is not None:
            self._listener()

    cdef void _append(self, bint x):
        cdef bint *mem

        if self._n_items == self._size:
            mem = <bint *>PyMem_Realloc(self._data, 2 * self._size * sizeof(bint))
            if not mem:
                raise MemoryError()

            self._data = mem
            self._size *= 2

        self._data[self._n_items] = x
        self._n_items += 1

