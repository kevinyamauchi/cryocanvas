import napari

# from cellcanvas.instance._qt.qt_segment_manager import QtSegmentManager
from cellcanvas._app.main_app import CellCanvasApp, QtCellCanvas
from cellcanvas.data.data_manager import DataManager
from cellcanvas.data.data_set import DataSet

# load the image
# with h5py.File("covid_crop_embedding.h5", "r") as f:
#     image = f["image"][:]
#
# # load the segmentation
# segmentation = mrcfile.read("TS_004_dose-filt_lp50_bin8_membrain_model.ckpt_segmented.mrc").astype(int)
#
# viewer = napari.Viewer()
# image_layer = viewer.add_image(image)
# labels_layer = viewer.add_labels(segmentation[0:100, 125:325, 125:375])

base_file_path = "./seg/cropped_covid.zarr"

dataset = DataSet.from_paths(
    image_path=f"{base_file_path}/image/cropped_covid",
    features_path=f"{base_file_path}/embedding/swin_unetr",
    labels_path=f"{base_file_path}/painting",
    segmentation_path=f"{base_file_path}/prediction",
    make_missing_datasets=True,
)

data_manager = DataManager(datasets=[dataset])
viewer = napari.Viewer()
cell_canvas = CellCanvasApp(data=data_manager, viewer=viewer, verbose=True)
cell_canvas_widget = QtCellCanvas(app=cell_canvas)
viewer.window.add_dock_widget(cell_canvas_widget)

napari.run()
