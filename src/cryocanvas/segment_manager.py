import numpy as np
from napari.layers import Labels
import pandas as pd

from cryocanvas.constants import PAINTABLE_KEY, CLASS_KEY, UNASSIGNED_CLASS
from cryocanvas.paint import paint as monkey_paint
from cryocanvas.fill import fill as monkey_fill


class SegmentManager:
    def __init__(self, labels_layer: Labels):
        self.labels_layer = labels_layer

        self._validate_features_table()

        # monkey patch our painting function
        self.labels_layer.paint = monkey_paint.__get__(self.labels_layer, Labels)
        self.labels_layer.fill = monkey_fill.__get__(self.labels_layer, Labels)

    def _validate_features_table(self):
        """Validate the features table in the labels layer.

        The features table must contain:
            - index (instance ID)
            - class membership
            - paintable
        """
        features_table = self.labels_layer.features

        if len(features_table) == 0:
            # if the table is empty, initialize it with the index
            label_values = np.unique(self.labels_layer.data)
            features_table = pd.DataFrame({"index": label_values})
        else:
            # verify the feature contains the index column
            if "index" not in features_table:
                raise ValueError("features table must contain `index` column with instance IDs.")

        # check if it has paintable attribute
        if PAINTABLE_KEY not in features_table:
            # default all are paintable
            features_table[PAINTABLE_KEY] = True

        # check if the features table has a class attribute
        if CLASS_KEY not in features_table:
            # default are all unassigned
            features_table[CLASS_KEY] = UNASSIGNED_CLASS

        # set the validated features
        self.labels_layer.features = features_table

    @property
    def paintable_labels(self) -> np.ndarray:
        features_table = self.labels_layer.features
        return features_table.loc[features_table[PAINTABLE_KEY]]["index"].values()
