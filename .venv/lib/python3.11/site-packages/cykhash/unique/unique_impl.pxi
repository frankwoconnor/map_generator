
"""
Template for sets

WARNING: DO NOT edit .pxi FILE directly, .pxi is generated from .pxi.in
"""




cpdef unique_int64(int64_t[:] vals, double size_hint=0.0):
    cdef Int64Set s = Int64Set_from_buffer(vals, size_hint)
    
    # compress:
    cdef int64_t* mem = s.table.keys
    cdef khint_t i
    cdef khint_t current = 0
    for i in range(s.table.n_buckets):
        if kh_exist_int64set(s.table, i):
            mem[current] = mem[i]
            current += 1

    # take over the memory:
    s.table.keys = NULL
    
    # shrink to fit:
    mem = <int64_t*> cykhash_traced_realloc(mem, sizeof(int64_t)*current);
    return MemoryNanny.create_memory_nanny(mem, current, sizeof(int64_t), b"q")


cpdef unique_stable_int64(int64_t[:] vals, double size_hint=0.0):
    # prepare 
    cdef Py_ssize_t n = len(vals)
    cdef Py_ssize_t at_least_needed = element_n_from_size_hint(<khint_t>n, size_hint)
    res=Int64Set(number_of_elements_hint=at_least_needed)
    cdef int64_t* mem = <int64_t*> cykhash_traced_malloc(sizeof(int64_t)*n);
    
    # insert
    cdef khint_t current = 0
    cdef Py_ssize_t i
    cdef int64_t element
    for i in range(n):
        element = vals[i]
        res.add(element)
        if current != res.size():
            mem[current] = element
            current += 1
    
    # shrink to fit:
    mem = <int64_t*> cykhash_traced_realloc(mem, sizeof(int64_t)*current);
    return MemoryNanny.create_memory_nanny(mem, current, sizeof(int64_t), b"q")

cpdef unique_float64(float64_t[:] vals, double size_hint=0.0):
    cdef Float64Set s = Float64Set_from_buffer(vals, size_hint)
    
    # compress:
    cdef float64_t* mem = s.table.keys
    cdef khint_t i
    cdef khint_t current = 0
    for i in range(s.table.n_buckets):
        if kh_exist_float64set(s.table, i):
            mem[current] = mem[i]
            current += 1

    # take over the memory:
    s.table.keys = NULL
    
    # shrink to fit:
    mem = <float64_t*> cykhash_traced_realloc(mem, sizeof(float64_t)*current);
    return MemoryNanny.create_memory_nanny(mem, current, sizeof(float64_t), b"d")


cpdef unique_stable_float64(float64_t[:] vals, double size_hint=0.0):
    # prepare 
    cdef Py_ssize_t n = len(vals)
    cdef Py_ssize_t at_least_needed = element_n_from_size_hint(<khint_t>n, size_hint)
    res=Float64Set(number_of_elements_hint=at_least_needed)
    cdef float64_t* mem = <float64_t*> cykhash_traced_malloc(sizeof(float64_t)*n);
    
    # insert
    cdef khint_t current = 0
    cdef Py_ssize_t i
    cdef float64_t element
    for i in range(n):
        element = vals[i]
        res.add(element)
        if current != res.size():
            mem[current] = element
            current += 1
    
    # shrink to fit:
    mem = <float64_t*> cykhash_traced_realloc(mem, sizeof(float64_t)*current);
    return MemoryNanny.create_memory_nanny(mem, current, sizeof(float64_t), b"d")

cpdef unique_int32(int32_t[:] vals, double size_hint=0.0):
    cdef Int32Set s = Int32Set_from_buffer(vals, size_hint)
    
    # compress:
    cdef int32_t* mem = s.table.keys
    cdef khint_t i
    cdef khint_t current = 0
    for i in range(s.table.n_buckets):
        if kh_exist_int32set(s.table, i):
            mem[current] = mem[i]
            current += 1

    # take over the memory:
    s.table.keys = NULL
    
    # shrink to fit:
    mem = <int32_t*> cykhash_traced_realloc(mem, sizeof(int32_t)*current);
    return MemoryNanny.create_memory_nanny(mem, current, sizeof(int32_t), b"i")


cpdef unique_stable_int32(int32_t[:] vals, double size_hint=0.0):
    # prepare 
    cdef Py_ssize_t n = len(vals)
    cdef Py_ssize_t at_least_needed = element_n_from_size_hint(<khint_t>n, size_hint)
    res=Int32Set(number_of_elements_hint=at_least_needed)
    cdef int32_t* mem = <int32_t*> cykhash_traced_malloc(sizeof(int32_t)*n);
    
    # insert
    cdef khint_t current = 0
    cdef Py_ssize_t i
    cdef int32_t element
    for i in range(n):
        element = vals[i]
        res.add(element)
        if current != res.size():
            mem[current] = element
            current += 1
    
    # shrink to fit:
    mem = <int32_t*> cykhash_traced_realloc(mem, sizeof(int32_t)*current);
    return MemoryNanny.create_memory_nanny(mem, current, sizeof(int32_t), b"i")

cpdef unique_float32(float32_t[:] vals, double size_hint=0.0):
    cdef Float32Set s = Float32Set_from_buffer(vals, size_hint)
    
    # compress:
    cdef float32_t* mem = s.table.keys
    cdef khint_t i
    cdef khint_t current = 0
    for i in range(s.table.n_buckets):
        if kh_exist_float32set(s.table, i):
            mem[current] = mem[i]
            current += 1

    # take over the memory:
    s.table.keys = NULL
    
    # shrink to fit:
    mem = <float32_t*> cykhash_traced_realloc(mem, sizeof(float32_t)*current);
    return MemoryNanny.create_memory_nanny(mem, current, sizeof(float32_t), b"f")


cpdef unique_stable_float32(float32_t[:] vals, double size_hint=0.0):
    # prepare 
    cdef Py_ssize_t n = len(vals)
    cdef Py_ssize_t at_least_needed = element_n_from_size_hint(<khint_t>n, size_hint)
    res=Float32Set(number_of_elements_hint=at_least_needed)
    cdef float32_t* mem = <float32_t*> cykhash_traced_malloc(sizeof(float32_t)*n);
    
    # insert
    cdef khint_t current = 0
    cdef Py_ssize_t i
    cdef float32_t element
    for i in range(n):
        element = vals[i]
        res.add(element)
        if current != res.size():
            mem[current] = element
            current += 1
    
    # shrink to fit:
    mem = <float32_t*> cykhash_traced_realloc(mem, sizeof(float32_t)*current);
    return MemoryNanny.create_memory_nanny(mem, current, sizeof(float32_t), b"f")
