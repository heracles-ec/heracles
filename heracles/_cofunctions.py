# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''cofunction: simple coroutines'''

__version__ = '2023.2.1'
__all__ = ['cofunction']

from functools import wraps, partial
from weakref import proxy, finalize


def _cofunction_finish(generator):
    try:
        next(generator)
    except StopIteration as end:
        return end.value
    else:
        raise RuntimeError('cofunction did not stop on finish()')


def cofunction(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        generator = function(*args, **kwargs)
        cofunction = next(generator)
        cofunction.finish = partial(_cofunction_finish, generator)
        finalize(cofunction, generator.close)
        return proxy(cofunction)
    return wrapper
