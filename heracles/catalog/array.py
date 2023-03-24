'''module for array catalogues'''

from .base import CatalogBase, CatalogPage


class ArrayCatalog(CatalogBase):
    '''catalogue reader for arrays'''

    def __init__(self, arr):
        '''create a new array catalogue reader'''
        super().__init__()
        self._arr = arr

    def __copy__(self):
        '''return a copy of this catalogue'''
        other = super().__copy__()
        other._arr = self._arr
        return other

    def _names(self):
        return self._arr.dtype.names

    def _size(self, selection):
        if selection is None:
            return len(self._arr)
        else:
            return len(self._arr[selection])

    def _join(self, first, *other):
        '''join boolean masks'''
        mask = first
        for a in other:
            mask = mask & a
        return mask

    def _pages(self, selection):
        '''iterate the rows of the array in pages'''
        if selection is None:
            arr = self._arr
        else:
            arr = self._arr[selection]
        nrows = len(arr)
        page_size = self.page_size
        names = arr.dtype.names
        for i in range(0, nrows, page_size):
            page = {name: arr[name][i:i+page_size] for name in names}
            yield CatalogPage(page)
