from typing import List, Optional, Tuple

import numpy as np
from napari.utils.events.containers import SelectableEventedList
from zarr import Array

from cellcanvas.data.data_set import DataSet


class DataManager:
    def __init__(self, datasets: Optional[List[DataSet]] = None):
        if datasets is None:
            datasets = []
        elif isinstance(datasets, DataSet):
            datasets = [datasets]
        self.datasets = SelectableEventedList(datasets)

    def get_training_data(self) -> Tuple[Array, Array]:
        """Get the pixel-wise semantic segmentation training data for datasets.

        Returns
        -------
        features : Array
            (n, d) array of features where n is the number of pixels
            and d is the number feature dimensions.
        labels : Array
            (n,) array of labels for each feature.
        """

        features = []
        labels = []
        for dataset in self.datasets:
            # get the features and labels
            # todo make lazier
            dataset_features = np.moveaxis(np.asarray(dataset.features), 0, -1)
            dataset_labels = np.asarray(dataset.labels)

            # reshape the data
            dataset_labels = dataset_labels.flatten()
            reshaped_features = dataset_features.reshape(
                -1, dataset_features.shape[-1]
            )

            # Filter features where labels are greater than 0
            valid_labels = dataset_labels > 0
            filtered_features = reshaped_features[valid_labels, :]
            filtered_labels = dataset_labels[valid_labels] - 1  # Adjust labels

            features.append(filtered_features)
            labels.append(filtered_labels)

        return np.concatenate(features), np.concatenate(labels)
