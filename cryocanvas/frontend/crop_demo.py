import logging
import sys
import zarr
import numpy as np
import napari
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QCheckBox, QGroupBox
from skimage import future
from sklearn.ensemble import RandomForestClassifier
from functools import partial
from psygnal import debounced
from superqt import ensure_main_thread
import threading

# Set up logging
def setup_logging():
    logger = logging.getLogger("cryocanvas")
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

model = None

def threaded_on_data_change(
    viewer,
    data_layer,
    painting_layer,
    prediction_layer,
    model_type,
    feature_params,
    live_fit,
    live_prediction,
    data_choice,
    logger,
    feature_data_skimage,
    feature_data_tomotwin
):
    global model
    logger.info("Labels data has changed!")

    # Determine which features to use based on the UI settings
    if feature_params['texture']:
        feature_data = feature_data_skimage
    else:
        # Example fallback if other feature sets are preferred
        feature_data = feature_data_tomotwin

    mask_idx = None
    training_features = None
    training_labels = None

    # Decide on the region of interest based on the widget selection
    if data_choice == "Current Displayed Region":
        corner_pixels = viewer.layers['Image'].corner_pixels
        mask_idx = (slice(None), slice(corner_pixels[0, 1], corner_pixels[1, 1] + 1), slice(corner_pixels[0, 2], corner_pixels[1, 2] + 1))
        training_features = feature_data[mask_idx]
        training_labels = painting_layer.data[mask_idx]
    elif data_choice == "Whole Image":
        training_features = feature_data[:]
        training_labels = painting_layer.data[:]        

    if training_features is None or training_labels is None or not np.any(training_labels > 0):
        logger.info("No valid training data. Skipping model update.")
        return

    print(f"Mask indices {mask_idx}")

    # Flatten labels for training
    # training_labels = training_labels.flatten()

    # Fit the model if live fitting is enabled
    if live_fit:
        model = train_and_predict(training_features, training_labels, model_type)

    # Predict if live prediction is enabled
    if live_prediction and model is not None:
        prediction = predict(model, training_features, model_type)
        print(f"Created a prediction with shape {prediction.shape} from features of shape {training_features.shape}")
        # Ensure thread safety when updating the prediction layer
        @ensure_main_thread
        def update_prediction():
            prediction_layer.data = prediction.reshape(painting_layer.data.shape)
        update_prediction()


def update_model(labels, features, model_type):
    # Flatten labels
    labels = labels.flatten()

    # Flatten the first three dimensions of features
    # New shape will be (10*200*200, 32)
    reshaped_features = features.reshape(-1, features.shape[-1])

    # Filter features where labels are greater than 0
    valid_labels = labels > 0
    filtered_features = reshaped_features[valid_labels, :]
    filtered_labels = labels[valid_labels] - 1  # Adjust labels

    # Model fitting
    if model_type == "Random Forest":
        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)
        clf.fit(filtered_features, filtered_labels)
        return clf
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def predict(model, features, model_type):
    # We shift labels + 1 because background is 0 and has special meaning
    prediction = future.predict_segmenter(features.reshape(-1, features.shape[-1]), model).reshape(features.shape[:-1]) + 1

    return np.transpose(prediction)

    
def connect_event_listeners(viewer, widget, logger, feature_data_skimage, feature_data_tomotwin):
    def on_data_change(event, viewer=viewer, widget=widget, logger=logger, feature_data_skimage=feature_data_skimage, feature_data_tomotwin=feature_data_tomotwin):
        # Extract necessary layers
        data_layer = viewer.layers['Image']
        painting_layer = viewer.layers['Painting']
        prediction_layer = viewer.layers['Prediction']

        # Call threaded function with all necessary arguments
        thread = threading.Thread(
            target=threaded_on_data_change,
            args=(
                viewer,
                data_layer,
                painting_layer,
                prediction_layer,
                widget.model_dropdown.currentText(),
                {
                    "intensity": widget.intensity_checkbox.isChecked(),
                    "edges": widget.edges_checkbox.isChecked(),
                    "texture": widget.texture_checkbox.isChecked(),
                },
                widget.live_fit_checkbox.isChecked(),
                widget.live_pred_checkbox.isChecked(),
                widget.data_dropdown.currentText(),
                logger,
                feature_data_skimage,
                feature_data_tomotwin
            ),
        )
        thread.start()
    
    debounced_listener = debounced(
        ensure_main_thread(on_data_change), timeout=1000
    )

    # Connect to relevant Napari viewer events
    viewer.camera.events.connect(debounced_listener)
    viewer.dims.events.connect(debounced_listener)
    viewer.layers['Painting'].events.set_data.connect(debounced_listener)

        
# Widget for the UI components
class NapariMLWidget(QWidget):
    def __init__(self, parent=None):
        super(NapariMLWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)
        self.setupModelSelection()
        self.setupFeatureSelection()
        self.setupDataSelection()
        self.setupLiveOptions()

    def setupModelSelection(self):
        model_label = QLabel("Select Model")
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["Random Forest"])
        model_layout = QHBoxLayout()
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_dropdown)
        self.layout.addLayout(model_layout)

    def setupFeatureSelection(self):
        self.intensity_checkbox = QCheckBox("Basic")
        self.edges_checkbox = QCheckBox("CNN features")
        self.texture_checkbox = QCheckBox("Embedding")
        self.texture_checkbox.setChecked(True)

        features_group = QGroupBox("Features")
        features_layout = QVBoxLayout()
        features_layout.addWidget(self.intensity_checkbox)
        features_layout.addWidget(self.edges_checkbox)
        features_layout.addWidget(self.texture_checkbox)
        features_group.setLayout(features_layout)
        self.layout.addWidget(features_group)

    def setupDataSelection(self):
        data_label = QLabel("Select Data for Model Fitting")
        self.data_dropdown = QComboBox()
        self.data_dropdown.addItems(["Current Displayed Region", "Whole Image"])
        data_layout = QHBoxLayout()
        data_layout.addWidget(data_label)
        data_layout.addWidget(self.data_dropdown)
        self.layout.addLayout(data_layout)

    def setupLiveOptions(self):
        self.live_fit_checkbox = QCheckBox("Live Model Fitting")
        self.live_fit_checkbox.setChecked(True)
        self.layout.addWidget(self.live_fit_checkbox)

        self.live_pred_checkbox = QCheckBox("Live Prediction")
        self.live_pred_checkbox.setChecked(True)
        self.layout.addWidget(self.live_pred_checkbox)


def train_and_predict(features, labels, model_type):
    # Assuming labels are already in a flattened form suitable for model training
    # Filtering out background or unlabelled data
    # import pdb; pdb.set_trace()
    train_features = features[labels > 0, :]
    train_labels = labels[labels > 0]

    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=50, random_state=0)
        model.fit(train_features, train_labels)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

# Main function to run the application
def main(zarr_path):
    logger = setup_logging()
    
    # Open the Zarr file and expect a specific structure
    dataset = zarr.open(zarr_path, mode='r')
    image_data = dataset['crop/original_data']
    feature_data_skimage = dataset['features/skimage']
    feature_data_tomotwin = dataset['features/tomotwin']

    viewer = napari.Viewer()
    data_layer = viewer.add_image(image_data[:], name="Image")

    # Create a prediction layer
    prediction_data = zarr.open(
        f"{zarr_path}/prediction",
        mode='a',
        shape=image_data.shape,
        dtype='i4',
        dimension_separator=".",
    )
    viewer.add_labels(prediction_data, name="Prediction", scale=data_layer.scale)

    # Create a painting layer
    painting_data = zarr.open(
        f"{zarr_path}/painting",
        mode='a',
        shape=image_data.shape,
        dtype='i4',
        dimension_separator=".",
    )
    viewer.add_labels(painting_data, name="Painting", scale=data_layer.scale)

    widget = NapariMLWidget()
    viewer.window.add_dock_widget(widget, name="CryoCanvas")

    # Connect event listeners
    connect_event_listeners(viewer, widget, logger, feature_data_skimage, feature_data_tomotwin)


if __name__ == "__main__":
    zarr_path = "/Users/kharrington/Data/CryoCanvas/cryocanvas_crop_005.zarr"
    main(zarr_path)

