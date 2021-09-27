from .functools import p, d

def tfunctag(tag=None):
    def _tfunc_decorator(func):
        def run(*argv, **kargs):
            # p("--------------------------------------------")
            p(f"[Trans Utils] Function Name: {func.__name__}")
            if tag is not None: p(f"[Trans Utils] Ps: {tag}")
            ret = func(*argv, **kargs)
            return ret
        return run
    return _tfunc_decorator


def tfuncname(func):
    def run(*argv, **kargs):
        p(f"[Trans Utils] Function Name: {func.__name__}")
        ret = func(*argv, **kargs)
        return ret
    return run
