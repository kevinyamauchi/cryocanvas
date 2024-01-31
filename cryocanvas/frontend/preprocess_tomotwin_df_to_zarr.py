from appdirs import user_data_dir
import os
import zarr
import dask.array as da
import toolz as tz

from sklearn.ensemble import RandomForestClassifier

from skimage import data, segmentation, feature, future
from skimage.feature import multiscale_basic_features
from skimage.io import imread, imshow
import numpy as np
from functools import partial
import napari
import threading

from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

from functools import partial
from psygnal import debounced
from superqt import ensure_main_thread

import logging
import sys

LOGGER = logging.getLogger("cryocanvas")
LOGGER.setLevel(logging.DEBUG)

streamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
streamHandler.setFormatter(formatter)
LOGGER.addHandler(streamHandler)

from scipy.spatial.distance import euclidean
import mrcfile

import pandas as pd
from numba import njit
from zarr.storage import init_array, init_group, BaseStore
from zarr.util import json_dumps

from scipy.spatial import cKDTree

from numcodecs import Blosc

base_dir = "/Users/kharrington/Data/CZII/4starsandhigher_20231230_194623"

mrc_path = f"{base_dir}/Position_44_SIRT_rec.mrc"
embeddings_path = f"{base_dir}/Position_44_SIRT_rec_embeddings.temb"
umap_path = f"{base_dir}/Position_44_SIRT_rec_embeddings_v2.tumap"

viewer = napari.Viewer()

# X, Y, Z, 0-32 features
embedding_df = pd.read_pickle(embeddings_path)
#umap = pd.read_pickle(umap_path)

mrc_data = mrcfile.mmap(mrc_path, permissive=True).data

data_layer = viewer.add_image(mrc_data)

image_shape = mrc_data.shape


# output zarr


z_range = slice(450, 850)
y_range = slice(300, 600)
x_range = slice(300, 600)

# Step 1: Filter embedding_df for the crop range
filtered_df = embedding_df[(embedding_df['Z'].between(z_range.start, z_range.stop - 1)) &
                           (embedding_df['Y'].between(y_range.start, y_range.stop - 1)) &
                           (embedding_df['X'].between(x_range.start, x_range.stop - 1))]

# Step 2: Create the crop from mrc_data
crop = mrc_data[z_range, y_range, x_range]

# Step 3: Create an array for the crop embeddings
embedding_dims = 32
embedding_array = np.zeros((crop.shape[0], crop.shape[1], crop.shape[2], embedding_dims))

# Step 4: Populate the embedding array
for index, row in filtered_df.iterrows():
    z, y, x = int(row['Z']) - z_range.start, int(row['Y']) - y_range.start, int(row['X']) - x_range.start
    embedding_array[z, y, x, :] = row.values[3:]

# Step 5: Write to Zarr file
with zarr.open('my_data.zarr', mode='w') as zarr_file:
    zarr_file.create_dataset('crop', data=crop, chunks=(10, 200, 200))
    zarr_file.create_dataset('embedding', data=embedding_array, chunks=(10, 200, 200, embedding_dims))

# Access the datasets from the Zarr file
crop_data = zarr_file['crop'][:]
embedding_data = np.transpose(zarr_file['embedding'][:], (3, 0, 1, 2))

# Add the datasets to the viewer
viewer.add_image(crop_data, name='crop')
viewer.add_image(embedding_data, name='embedding')
