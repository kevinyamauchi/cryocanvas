from scipy import ndimage as ndi
from napari.layers.labels.labels import _coerce_indices_for_vectorization
import numpy as np

from cellcanvas.constants import PAINTABLE_KEY


def fill(self, coord, new_label, refresh=True):
        """Replace an existing label with a new label, either just at the
        connected component if the `contiguous` flag is `True` or everywhere
        if it is `False`, working in the number of dimensions specified by
        the `n_edit_dimensions` flag.

        This has been modified to only fill connected components that are
        annotated as "paintable".

        Parameters
        ----------
        coord : sequence of float
            Position of mouse cursor in image coordinates.
        new_label : int
            Value of the new label to be filled in.
        refresh : bool
            Whether to refresh view slice or not. Set to False to batch paint
            calls.
        """
        int_coord = tuple(np.round(coord).astype(int))
        # If requested fill location is outside data shape then return
        if np.any(np.less(int_coord, 0)) or np.any(
            np.greater_equal(int_coord, self.data.shape)
        ):
            return

        # If requested new label doesn't change old label then return
        # If preserve_labels is True, then only paint over paintable labels
        old_label = np.asarray(self.data[int_coord]).item()
        # get the paintable label values
        features_table = self.features
        paintable_values = features_table.loc[features_table[PAINTABLE_KEY]]["index"].values
        if old_label == new_label or (
            self.preserve_labels
            and old_label not in paintable_values 
        ):
            return

        dims_to_fill = sorted(
            self._slice_input.order[-self.n_edit_dimensions :]
        )
        data_slice_list = list(int_coord)
        for dim in dims_to_fill:
            data_slice_list[dim] = slice(None)
        data_slice = tuple(data_slice_list)
        labels = np.asarray(self.data[data_slice])
        slice_coord = tuple(int_coord[d] for d in dims_to_fill)

        matches = labels == old_label
        if self.contiguous:
            # if contiguous replace only selected connected component
            labeled_matches, num_features = ndi.label(matches)
            if num_features != 1:
                match_label = labeled_matches[slice_coord]
                matches = np.logical_and(
                    matches, labeled_matches == match_label
                )

        match_indices_local = np.nonzero(matches)
        if self.ndim not in {2, self.n_edit_dimensions}:
            n_idx = len(match_indices_local[0])
            match_indices = []
            j = 0
            for d in data_slice:
                if isinstance(d, slice):
                    match_indices.append(match_indices_local[j])
                    j += 1
                else:
                    match_indices.append(np.full(n_idx, d, dtype=np.intp))
        else:
            match_indices = match_indices_local

        match_indices = _coerce_indices_for_vectorization(
            self.data, match_indices
        )

        self.data_setitem(match_indices, new_label, refresh)