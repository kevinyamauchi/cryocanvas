import pandas as pd
from napari.layers import Labels
import numpy as np
import pytest

from cryocanvas.segment_manager import SegmentManager
from cryocanvas.constants import PAINTABLE_KEY, CLASS_KEY, UNASSIGNED_CLASS


def _make_labels_layer() -> Labels:
    label_image = np.zeros((10, 10, 10), dtype=int)
    label_image[0:5, 0:5, 0:5] = 1
    label_image[5:10, 5:10, 5:10] = 2
    return Labels(label_image)


def test_feature_validation_no_features_table():
    """Check that all necessary columns in the feature table are added
    when they are not initially present.
    """
    labels_layer = _make_labels_layer()

    segment_manager = SegmentManager(labels_layer)
    features_table = labels_layer.features

    # check that the index column was added
    np.testing.assert_array_equal(np.array([0, 1, 2]), features_table["index"])

    # check that all label values (0, 1, 2) are set to paintable
    np.testing.assert_array_equal(
        np.array([True, True, True]),
        features_table[PAINTABLE_KEY]
    )

    # check that the class was assigned as "unassigned"
    np.testing.assert_array_equal(
        np.array([UNASSIGNED_CLASS, UNASSIGNED_CLASS, UNASSIGNED_CLASS]),
        features_table[CLASS_KEY]
    )


def test_feature_validation_invalid_features_table():
    """Check that invalid features table raise an error."""
    labels_layer = _make_labels_layer()

    labels_layer.features = pd.DataFrame({"not_index": [0, 1, 2]})

    with pytest.raises(ValueError):
        # if there is a features table, it should have an "index" column
        # with the instance IDs
        _ = SegmentManager(labels_layer)


def test_feature_validation_valid_features_table():
    """Check that valid features table raise is not overwritten."""
    labels_layer = _make_labels_layer()

    paintable_values = [False, False, False]
    class_values = ["background", "hello", "hello"]
    labels_layer.features = pd.DataFrame(
        {
            "index": [0, 1, 2],
            PAINTABLE_KEY: paintable_values,
            CLASS_KEY: class_values,
        }

    )

    segment_manager = SegmentManager(labels_layer)
    features_table = labels_layer.features

    # check that the paintable values were not overwritten
    np.testing.assert_array_equal(
        paintable_values,
        features_table[PAINTABLE_KEY]
    )

    # check that the class was assigned were not overwritten
    np.testing.assert_array_equal(
        class_values,
        features_table[CLASS_KEY]
    )
