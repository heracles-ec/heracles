'''module for utilities'''


def toc_match(key, include=None, exclude=None):
    '''return whether a tocdict entry matches include/exclude criteria'''
    if include is not None:
        for pattern in include:
            if all(p is Ellipsis or p == k for p, k in zip(pattern, key)):
                break
        else:
            return False
    if exclude is not None:
        for pattern in exclude:
            if all(p is Ellipsis or p == k for p, k in zip(pattern, key)):
                return False
    return True
