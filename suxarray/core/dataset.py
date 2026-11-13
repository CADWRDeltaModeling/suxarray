""" suxarray module

`suxarray` is a module that extends the functionality of `uxarray` for the
SCHISM grid.
"""

from typing import Optional
import uxarray as ux
from suxarray.grid import Grid
from suxarray.core.dataarray import SxDataArray


class SxDataset(ux.UxDataset):
    __slots__ = ("_sxgrid",)

    def __init__(
        self,
        *args,
        sxgrid: Grid = None,
        source_datasets: Optional[str] = None,
        **kwargs,
    ):
        self._sxgrid = sxgrid
        super().__init__(
            *args, uxgrid=sxgrid, source_datasets=source_datasets, **kwargs
        )

    def __getitem__(self, key):
        value = super().__getitem__(key)

        if isinstance(value, ux.UxDataArray):
            value = SxDataArray(value, sxgrid=self.sxgrid)

        return value

    @property
    def sxgrid(self):
        return self._sxgrid

