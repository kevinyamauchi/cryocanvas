from dataclasses import dataclass

import zarr
from zarr import Array


@dataclass
class DataSet:
    """Container for the data relating to a single tomogram"""

    image: Array
    features: Array
    labels: Array
    segmentation: Array

    def __hash__(self):
        """Simple hash function, should updated in the future.

        Note: this might create issues if files are moved.
        """
        return hash(self.image.path)

    @classmethod
    def from_paths(
        cls,
        image_path: str,
        features_path: str,
        labels_path: str,
        segmentation_path: str,
    ):
        image = zarr.open(image_path, "r")
        features = zarr.open(features_path, "r")
        labels = zarr.open(labels_path, "a")
        segmentation = zarr.open(segmentation_path, "a")

        return cls(
            image=image,
            features=features,
            labels=labels,
            segmentation=segmentation,
        )
