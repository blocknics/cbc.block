import sys

class object_pool:
    """Manage a free-list of objects. The objects are automatically made
    available as soon as they are deleted by the caller. The assumption is that
    any operation is repeated a number of times (think iterative solvers), so
    that if N objects are needed simultaneously then soon N objects are needed
    again. Thus, objects managed by this pool are not deleted until the owning
    object (typically a Matrix) is deleted.
    """
    def __init__(self):
        self.all = []

    def add(self, obj):
        self.all.append(obj)

    def get(self):
        for obj in self.all:
            if sys.getrefcount(obj) == 3:
                # 3 references: self.all, obj, getrefcount() parameter
                return obj


def shared_vec_pool(func):
    """Decorator for create_vec, which creates a per-object pool of (memoized)
    returned vectors, shared for all dimensions. To be used only on objects
    where it is known that the row and columns are distributed equally.
    """
    def pooled_create_vec(self, dim=1):
        try:
            vec_pool = self._vec_pool
        except AttributeError:
            vec_pool = self._vec_pool = object_pool()
        vec = vec_pool.get()
        if vec is None:
            vec = func(self, dim)
            vec_pool.add(vec)
        return vec
    pooled_create_vec.__doc__ = func.__doc__
    return pooled_create_vec

def vec_pool(func):
    """Decorator for create_vec, which creates a per-object pool of (memoized)
    returned vectors per dimension.
    """
    def pooled_create_vec(self, dim=1):
        try:
            vec_pools = self._vec_pools
        except AttributeError:
            vec_pools = self._vec_pools = [object_pool(), object_pool()]
        vec_pool = vec_pools[dim]
        vec = vec_pool.get()
        if vec is None:
            vec = func(self, dim)
            vec_pool.add(vec)
        return vec
    pooled_create_vec.__doc__ = func.__doc__
    return pooled_create_vec

def store_args_ref(func):
    """Decorator for any function, which stores a reference to the arguments
    on the object. Used to force a Python-side reference, when the native-side
    reference isn't sufficient (or present)."""
    def store_args_and_pass(self, *args, **kwargs):
        self._vec_pool_args = (args, kwargs)
        return func(self, *args, **kwargs)
    store_args_and_pass.__doc__ = func.__doc__
    return store_args_and_pass
