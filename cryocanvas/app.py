import napari
import zarr
import numpy as np
import threading
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QHBoxLayout,
    QCheckBox,
    QGroupBox,
)
from skimage.feature import multiscale_basic_features
from sklearn.ensemble import RandomForestClassifier
from skimage import future
import toolz as tz
from psygnal import debounced
from superqt import ensure_main_thread
import logging
import sys


# Define a class to encapsulate the Napari viewer and related functionalities
class CryoCanvasApp:
    def __init__(self, zarr_path):
        self.zarr_path = zarr_path
        self.dataset = zarr.open(zarr_path, mode="r")
        self.image_data = self.dataset["crop/original_data"]
        self.feature_data_skimage = self.dataset["features/skimage"]
        self.feature_data_tomotwin = self.dataset["features/tomotwin"]
        self.viewer = napari.Viewer()
        self._init_viewer_layers()
        self._init_logging()
        self._add_widget()
        self.model = None

    def _init_viewer_layers(self):
        self.data_layer = self.viewer.add_image(self.image_data, name="Image")
        self.prediction_data = zarr.open(
            f"{self.zarr_path}/prediction",
            mode="a",
            shape=self.image_data.shape,
            dtype="i4",
            dimension_separator=".",
        )
        self.prediction_layer = self.viewer.add_labels(
            self.prediction_data, name="Prediction", scale=self.data_layer.scale
        )
        self.painting_data = zarr.open(
            f"{self.zarr_path}/painting",
            mode="a",
            shape=self.image_data.shape,
            dtype="i4",
            dimension_separator=".",
        )
        self.painting_layer = self.viewer.add_labels(
            self.painting_data, name="Painting", scale=self.data_layer.scale
        )

    def _init_logging(self):
        self.logger = logging.getLogger("cryocanvas")
        self.logger.setLevel(logging.DEBUG)
        streamHandler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        streamHandler.setFormatter(formatter)
        self.logger.addHandler(streamHandler)

    def _add_widget(self):
        self.widget = NapariMLWidget()
        self.viewer.window.add_dock_widget(self.widget, name="CryoCanvas")
        self._connect_events()

    def _connect_events(self):
        # Use a partial function to pass additional arguments to the event handler
        on_data_change_handler = tz.curry(self.on_data_change)(app=self)
        for listener in [
            self.viewer.camera.events,
            self.viewer.dims.events,
            self.painting_layer.events.paint,
        ]:
            listener.connect(
                debounced(
                    ensure_main_thread(on_data_change_handler),
                    timeout=1000,
                )
            )

    def get_data_layer(self):
        return self.viewer.layers["Image"]

    def get_prediction_layer(self):
        return self.viewer.layers["Prediction"]

    def get_painting_layer(self):
        return self.viewer.layers["Painting"]

    def on_data_change(self, event, app):
        # Define corner_pixels based on the current view or other logic
        corner_pixels = self.viewer.layers['Image'].corner_pixels

        # Start the thread with correct arguments
        thread = threading.Thread(
            target=self.threaded_on_data_change,
            args=(
                event,
                corner_pixels,
                self.viewer.dims.current_step,
                self.widget.model_dropdown.currentText(),
                {
                    "basic": self.widget.basic_checkbox.isChecked(),
                    "embedding": self.widget.embedding_checkbox.isChecked(),
                },
                self.widget.live_fit_checkbox.isChecked(),
                self.widget.live_pred_checkbox.isChecked(),
                self.widget.data_dropdown.currentText(),
            ),
        )
        thread.start()
        thread.join()

        # Ensure the prediction layer visual is updated
        self.get_prediction_layer().refresh()

    def threaded_on_data_change(
        self,
        event,
        corner_pixels,
        dims,
        model_type,
        feature_params,
        live_fit,
        live_prediction,
        data_choice,
    ):
        self.logger.info(f"Labels data has changed! {event}")

        # Find a mask of indices we will use for fetching our data
        mask_idx = (
            slice(
                self.viewer.dims.current_step[0], self.viewer.dims.current_step[0] + 1
            ),
            slice(corner_pixels[0, 1], corner_pixels[1, 1]),
            slice(corner_pixels[0, 2], corner_pixels[1, 2]),
        )
        if data_choice == "Whole Image":
            mask_idx = tuple([slice(0, sz) for sz in self.get_data_layer().data.shape])

        self.logger.info(
            f"mask idx {mask_idx}, image {self.get_data_layer().data.shape}"
        )
        active_image = self.get_data_layer().data[mask_idx]
        self.logger.info(
            f"active image shape {active_image.shape} data choice {data_choice} painting_data {self.painting_data.shape} mask_idx {mask_idx}"
        )

        active_labels = self.painting_data[mask_idx]

        def compute_features(mask_idx, use_skimage_features, use_tomotwin_features):
            features = []
            if use_skimage_features:
                features.append(
                    self.feature_data_skimage[mask_idx].reshape(
                        -1, self.feature_data_skimage.shape[-1]
                    )
                )
            if use_tomotwin_features:
                features.append(
                    self.feature_data_tomotwin[mask_idx].reshape(
                        -1, self.feature_data_tomotwin.shape[-1]
                    )
                )

            if features:
                return np.concatenate(features, axis=1)
            else:
                raise ValueError("No features selected for computation.")

        training_labels = None

        use_skimage_features = False
        use_tomotwin_features = True

        if data_choice == "Current Displayed Region":
            # Use only the currently displayed region.
            training_features = compute_features(active_image, mask_idx, use_skimage_features, use_tomotwin_features)
            training_labels = np.squeeze(active_labels)
        elif data_choice == "Whole Image":
            if use_skimage_features:
                training_features = np.array(self.feature_data_skimage)
            else:
                training_features = np.array(self.feature_data_tomotwin)
            training_labels = np.array(self.painting_data)
        else:
            raise ValueError(f"Invalid data choice: {data_choice}")

        if (training_labels is None) or np.any(training_labels.shape == 0):
            self.logger.info("No training data yet. Skipping model update")
        elif live_fit:
            # Retrain model
            self.logger.info(
                f"training model with labels {training_labels.shape} features {training_features.shape} unique labels {np.unique(training_labels[:])}"
            )
            self.model = self.update_model(training_labels, training_features, model_type)

        # Don't do live prediction on whole image, that happens earlier slicewise
        if live_prediction:
            # Update prediction_data
            if use_skimage_features:
                prediction_features = np.array(self.feature_data_skimage)
            else:
                prediction_features = np.array(self.feature_data_tomotwin)            
            # Add 1 becasue of the background label adjustment for the model
            prediction = self.predict(self.model, prediction_features, model_type)
            self.logger.info(
                f"prediction {prediction.shape} prediction layer {self.get_prediction_layer().data.shape} prediction {np.transpose(prediction).shape} features {prediction_features.shape}"
            )

            self.get_prediction_layer().data = np.transpose(prediction)

    def update_model(self, labels, features, model_type):
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
            clf = RandomForestClassifier(
                n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05
            )
            clf.fit(filtered_features, filtered_labels)
            return clf
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def predict(self, model, features, model_type):
        # We shift labels + 1 because background is 0 and has special meaning
        prediction = (
            future.predict_segmenter(
                features.reshape(-1, features.shape[-1]), model
            ).reshape(features.shape[:-1])
            + 1
        )

        return np.transpose(prediction)


class NapariMLWidget(QWidget):
    def __init__(self, parent=None):
        super(NapariMLWidget, self).__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Dropdown for selecting the model
        model_label = QLabel("Select Model")
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["Random Forest"])
        model_layout = QHBoxLayout()
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_dropdown)
        layout.addLayout(model_layout)

        # Boolean options for features
        self.basic_checkbox = QCheckBox("Basic")  # Renamed from intensity_checkbox to basic_checkbox
        self.basic_checkbox.setChecked(True)
        self.embedding_checkbox = QCheckBox("Embedding")  # Renamed from texture_checkbox to embedding_checkbox
        self.embedding_checkbox.setChecked(True)

        features_group = QGroupBox("Features")
        features_layout = QVBoxLayout()
        features_layout.addWidget(self.basic_checkbox)
        features_layout.addWidget(self.embedding_checkbox)
        features_group.setLayout(features_layout)
        layout.addWidget(features_group)

        # Dropdown for data selection
        data_label = QLabel("Select Data for Model Fitting")
        self.data_dropdown = QComboBox()
        self.data_dropdown.addItems(["Current Displayed Region", "Whole Image"])
        self.data_dropdown.setCurrentText("Whole Image")
        data_layout = QHBoxLayout()
        data_layout.addWidget(data_label)
        data_layout.addWidget(self.data_dropdown)
        layout.addLayout(data_layout)

        # Checkbox for live model fitting
        self.live_fit_checkbox = QCheckBox("Live Model Fitting")
        self.live_fit_checkbox.setChecked(True)
        layout.addWidget(self.live_fit_checkbox)

        # Checkbox for live prediction
        self.live_pred_checkbox = QCheckBox("Live Prediction")
        self.live_pred_checkbox.setChecked(True)
        layout.addWidget(self.live_pred_checkbox)

        self.setLayout(layout)



# Initialize your application
if __name__ == "__main__":
    zarr_path = "/Users/kharrington/Data/CryoCanvas/cryocanvas_crop_006.zarr"
    app = CryoCanvasApp(zarr_path)
    napari.run()
