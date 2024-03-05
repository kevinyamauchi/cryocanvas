from typing import List, Tuple

import numpy as np
from napari.layers import Labels
from napari.utils.events import EventedSet, Event
import pandas as pd
from psygnal.containers import EventedList

from cryocanvas.constants import PAINTABLE_KEY, CLASS_KEY, UNASSIGNED_CLASS
from cryocanvas.paint import paint as monkey_paint
from cryocanvas.fill import fill as monkey_fill


class SegmentManager:
    def __init__(self, labels_layer: Labels, classes: Tuple[str, ...] = (UNASSIGNED_CLASS,)):
        self.labels_layer = labels_layer
        self.classes = EventedList(classes)

        self._validate_features_table()
        self._validate_classes()

        # object to store the currently selected label
        self._selected_labels = EventedSet()

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

    def _validate_classes(self):
        """Validate the classes that can be assigned to segments."""
        if UNASSIGNED_CLASS not in self.classes:
            # ensure the unassigned class is present in classes.
            self.classes.append(UNASSIGNED_CLASS)

    @property
    def paintable_labels(self) -> np.ndarray:
        features_table = self.labels_layer.features
        return features_table.loc[features_table[PAINTABLE_KEY]]["index"].values()

    def set_paintable_by_class(self, class_name: str, paintable: bool):
        """Set the paintable value for all instances of a given class."""
        pass

    def set_paintable_by_instance(self, label_value: str, paintable: bool):
        """Set all the paintable value for an instance"""
        pass

    def color_by_class(self):
        """Color segments by their class."""
        pass

    def color_by_instance(self):
        """Color segments by their instance ID."""
        pass

    def _connect_mouse_events(self, layer: Labels):
        layer.mouse_drag_callbacks.append(self._on_click_selection)

    def _on_click_selection(self, layer: Labels, event: Event):
        """Mouse callback for selecting labels"""
        label_index = layer.get_value(
            position=event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )
        selection_set = self._selected_labels
        print(event.modfiers)
        if "control" not in event.modifiers():
            # user must press control
            return

        if (label_index is None) or (label_index == layer._background_label):
            # the background or outside the layer was clicked, clear the selection
            if "shift" not in event.modifiers:
                # don't clear the selection if the shift key was held
                selection_set.clear()
            return
        if "shift" in event.modifiers:
            selection_set.symmetric_difference_update([label_index])
        else:
            selection_set._set.clear()
            selection_set.update([label_index])
