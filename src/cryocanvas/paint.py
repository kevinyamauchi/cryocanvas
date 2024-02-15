from napari.layers.labels._labels_utils import (
    indices_in_shape,
    sphere_indices,
)
from napari.layers.labels.labels import _coerce_indices_for_vectorization
import numpy as np

from cryocanvas.constants import PAINTABLE_KEY


def paint(self, coord, new_label, refresh=True):
    """Paint over existing labels with a new label, using the selected
    brush shape and size, either only on the visible slice or in all
    n dimensions.

    Parameters
    ----------
    coord : sequence of int
        Position of mouse cursor in image coordinates.
    new_label : int
        Value of the new label to be filled in.
    refresh : bool
        Whether to refresh view slice or not. Set to False to batch paint
        calls.
    """
    shape = self.data.shape
    dims_to_paint = sorted(
        self._slice_input.order[-self.n_edit_dimensions:]
    )
    dims_not_painted = sorted(
        self._slice_input.order[: -self.n_edit_dimensions]
    )
    paint_scale = np.array(
        [self.scale[i] for i in dims_to_paint], dtype=float
    )

    slice_coord = [int(np.round(c)) for c in coord]
    if self.n_edit_dimensions < self.ndim:
        coord_paint = [coord[i] for i in dims_to_paint]
        shape = [shape[i] for i in dims_to_paint]
    else:
        coord_paint = coord

    # Ensure circle doesn't have spurious point
    # on edge by keeping radius as ##.5
    radius = np.floor(self.brush_size / 2) + 0.5
    mask_indices = sphere_indices(radius, tuple(paint_scale))

    mask_indices = mask_indices + np.round(np.array(coord_paint)).astype(
        int
    )

    # discard candidate coordinates that are out of bounds
    mask_indices = indices_in_shape(mask_indices, shape)

    # Transfer valid coordinates to slice_coord,
    # or expand coordinate if 3rd dim in 2D image
    slice_coord_temp = list(mask_indices.T)
    if self.n_edit_dimensions < self.ndim:
        for j, i in enumerate(dims_to_paint):
            slice_coord[i] = slice_coord_temp[j]
        for i in dims_not_painted:
            slice_coord[i] = slice_coord[i] * np.ones(
                mask_indices.shape[0], dtype=int
            )
    else:
        slice_coord = slice_coord_temp

    slice_coord = _coerce_indices_for_vectorization(self.data, slice_coord)

    # slice coord is a tuple of coordinate arrays per dimension
    # subset it if we want to only paint into background/only erase
    # current label
    if self.preserve_labels:
        if new_label == self.colormap.background_value:
            keep_coords = self.data[slice_coord] == self.selected_label
        else:
            # get the paintable label values
            features_table = self.features
            paintable_values = features_table.loc[features_table[PAINTABLE_KEY]]["index"].values

            # may be slow, might need to update in the future
            keep_coords = np.isin(self.data[slice_coord], paintable_values)
        slice_coord = tuple(sc[keep_coords] for sc in slice_coord)

    self.data_setitem(slice_coord, new_label, refresh)