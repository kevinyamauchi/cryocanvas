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
    QPushButton,
    QSlider,
)
from qtpy.QtCore import Qt
from skimage.feature import multiscale_basic_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from skimage import future
import toolz as tz
from psygnal import debounced
from superqt import ensure_main_thread
import logging
import sys
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from magicgui.tqdm import tqdm
from napari.qt.threading import thread_worker
from sklearn.cross_decomposition import PLSRegression
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from threading import Thread

# from https://github.com/napari/napari/issues/4384

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
        self.create_embedding_plot()

    def get_labels_colormap(self):
        """Return a colormap for distinct label colors based on:
        Green-Armytage, P., 2010. A colour alphabet and the limits of colour coding. JAIC-Journal of the International Colour Association, 5.
        """
        colormap_22 = {
            0: np.array([0, 0, 0, 0]),  # alpha
            1: np.array([1, 1, 0, 1]),  # yellow
            2: np.array([0.5, 0, 0.5, 1]),  # purple
            3: np.array([1, 0.65, 0, 1]),  # orange
            4: np.array([0.68, 0.85, 0.9, 1]),  # light blue
            5: np.array([1, 0, 0, 1]),  # red
            6: np.array([1, 0.94, 0.8, 1]),  # buff
            7: np.array([0.5, 0.5, 0.5, 1]),  # grey
            8: np.array([0, 0.5, 0, 1]),  # green
            9: np.array([0.8, 0.6, 0.8, 1]),  # purplish pink
            10: np.array([0, 0, 1, 1]),  # blue
            11: np.array([1, 0.85, 0.7, 1]),  # yellowish pink
            12: np.array([0.54, 0.17, 0.89, 1]),  # violet
            13: np.array([1, 0.85, 0, 1]),  # orange yellow
            14: np.array([0.65, 0.09, 0.28, 1]),  # purplish red
            15: np.array([0.68, 0.8, 0.18, 1]),  # greenish yellow
            16: np.array([0.65, 0.16, 0.16, 1]),  # reddish brown
            17: np.array([0.5, 0.8, 0, 1]),  # yellow green
            18: np.array([0.8, 0.6, 0.2, 1]),  # yellowish brown
            19: np.array([1, 0.27, 0, 1]),  # reddish orange
            20: np.array([0.5, 0.5, 0.2, 1]),  # olive green
        }
        return colormap_22

    def _init_viewer_layers(self):
        self.data_layer = self.viewer.add_image(self.image_data, name="Image", projection_mode='mean')
        self.prediction_data = zarr.open(
            f"{self.zarr_path}/prediction",
            mode="a",
            shape=self.image_data.shape,
            dtype="i4",
            dimension_separator=".",
        )
        self.prediction_layer = self.viewer.add_labels(
            self.prediction_data,
            name="Prediction",
            scale=self.data_layer.scale,
            opacity=0.1,
            color=self.get_labels_colormap(),
        )
        self.painting_data = zarr.open(
            f"{self.zarr_path}/painting",
            mode="a",
            shape=self.image_data.shape,
            dtype="i4",
            dimension_separator=".",
        )
        self.painting_layer = self.viewer.add_labels(
            self.painting_data,
            name="Painting",
            scale=self.data_layer.scale,
            color=self.get_labels_colormap(),
        )

        # Set defaults for layers
        self.get_painting_layer().brush_size = 2
        self.get_painting_layer().n_edit_dimensions = 3

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
        self.widget = CryoCanvasWidget(self)        
        self.viewer.window.add_dock_widget(self.widget, name="CryoCanvas")
        self.widget.estimate_background_button.clicked.connect(
            self.estimate_background
        )
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
        corner_pixels = self.viewer.layers["Image"].corner_pixels

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

        # Update class distribution charts
        self.update_class_distribution_charts()

        # Update projection
        self.create_embedding_plot()

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
                self.viewer.dims.current_step[0],
                self.viewer.dims.current_step[0] + 1,
            ),
            slice(corner_pixels[0, 1], corner_pixels[1, 1]),
            slice(corner_pixels[0, 2], corner_pixels[1, 2]),
        )
        if data_choice == "Whole Image":
            mask_idx = tuple(
                [slice(0, sz) for sz in self.get_data_layer().data.shape]
            )

        self.logger.info(
            f"mask idx {mask_idx}, image {self.get_data_layer().data.shape}"
        )
        active_image = self.get_data_layer().data[mask_idx]
        self.logger.info(
            f"active image shape {active_image.shape} data choice {data_choice} painting_data {self.painting_data.shape} mask_idx {mask_idx}"
        )

        active_labels = self.painting_data[mask_idx]

        def compute_features(
            mask_idx, use_skimage_features, use_tomotwin_features
        ):
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
            training_features = compute_features(
                mask_idx, use_skimage_features, use_tomotwin_features
            )
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
            self.model = self.update_model(
                training_labels, training_features, model_type
            )

        if live_prediction and self.model:
            # Update prediction_data
            if use_skimage_features:
                prediction_features = np.array(self.feature_data_skimage)
            else:
                prediction_features = np.array(self.feature_data_tomotwin)
            # Add 1 becasue of the background label adjustment for the model
            prediction = self.predict(
                self.model, prediction_features, model_type
            )
            self.logger.info(
                f"prediction {prediction.shape} prediction layer {self.get_prediction_layer().data.shape} prediction {np.transpose(prediction).shape} features {prediction_features.shape}"
            )

            # self.compute_and_visualize_2d_projection(prediction_features)
            
            self.get_prediction_layer().data = np.transpose(prediction)

    def update_model(self, labels, features, model_type):
        # Flatten labels
        labels = labels.flatten()
        reshaped_features = features.reshape(-1, features.shape[-1])

        # Filter features where labels are greater than 0
        valid_labels = labels > 0
        filtered_features = reshaped_features[valid_labels, :]
        filtered_labels = labels[valid_labels] - 1  # Adjust labels

        if filtered_labels.size == 0:
            self.logger.info("No labels present. Skipping model update.")
            return None

        # Calculate class weights
        unique_labels = np.unique(filtered_labels)
        class_weights = compute_class_weight(
            "balanced", classes=unique_labels, y=filtered_labels
        )
        weight_dict = dict(zip(unique_labels, class_weights))

        # Apply weights
        sample_weights = np.vectorize(weight_dict.get)(filtered_labels)

        # Model fitting
        if model_type == "Random Forest":
            clf = RandomForestClassifier(
                n_estimators=50,
                n_jobs=-1,
                max_depth=10,
                max_samples=0.05,
                class_weight=weight_dict,
            )
            clf.fit(filtered_features, filtered_labels)
            return clf
        elif model_type == "XGBoost":
            clf = xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1, use_label_encoder=False
            )
            clf.fit(
                filtered_features,
                filtered_labels,
                sample_weight=sample_weights,
            )
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

    def update_class_distribution_charts(self):
        total_pixels = np.product(self.painting_data.shape)

        painting_labels, painting_counts = np.unique(self.painting_data[:], return_counts=True)
        prediction_labels, prediction_counts = np.unique(self.get_prediction_layer().data[:], return_counts=True)

        # Calculate percentages instead of raw counts
        painting_percentages = (painting_counts / total_pixels) * 100
        prediction_percentages = (prediction_counts / total_pixels) * 100

        # Separate subplot for class 0 in painting layer
        unpainted_percentage = painting_percentages[painting_labels == 0] if 0 in painting_labels else [0]

        # Exclude class 0 for prediction layer
        valid_prediction_indices = prediction_labels > 0
        valid_prediction_labels = prediction_labels[valid_prediction_indices]
        valid_prediction_percentages = prediction_percentages[valid_prediction_indices]

        # Exclude class 0 for painting layer percentages
        valid_painting_indices = painting_labels > 0
        valid_painting_labels = painting_labels[valid_painting_indices]
        valid_painting_percentages = painting_percentages[valid_painting_indices]

        # Example class to color mapping
        class_color_mapping = {
            label: "#{:02x}{:02x}{:02x}".format(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
            for label, rgba in self.get_labels_colormap().items()
        }

        self.widget.figure.clear()

        napari_charcoal_hex = "#262930"

        # Custom style adjustments for dark theme
        dark_background_style = {
            "figure.facecolor": napari_charcoal_hex,
            "axes.facecolor": napari_charcoal_hex,
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "text.color": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }

        with plt.style.context(dark_background_style):
            # Create subplots with adjusted heights
            gs = self.widget.figure.add_gridspec(3, 1, height_ratios=[1, 4, 4])
            ax0 = self.widget.figure.add_subplot(gs[0])
            ax1 = self.widget.figure.add_subplot(gs[1])
            ax2 = self.widget.figure.add_subplot(gs[2])

            # Plot for unpainted pixels
            ax0.barh(0, unpainted_percentage, color="#AAAAAA", edgecolor="white")
            ax0.set_title("Unpainted Pixels")
            ax0.set_xlabel("% of Image")
            ax0.set_yticks([])  # Hide y-ticks for simplicity

            # Horizontal bar plots for painting and prediction layers
            ax1.barh(valid_painting_labels, valid_painting_percentages, color=[class_color_mapping.get(x, "#FFFFFF") for x in valid_painting_labels], edgecolor="white")
            ax1.set_title("Painting Layer")

            ax1.set_xlabel("% of Image")
            ax1.set_yticks(valid_painting_labels)
            ax1.invert_yaxis()  # Invert y-axis to have labels in ascending order from top to bottom

            ax2.barh(valid_prediction_labels, valid_prediction_percentages, color=[class_color_mapping.get(x, "#FFFFFF") for x in valid_prediction_labels], edgecolor="white")
            ax2.set_title("Prediction Layer")
            ax2.set_xlabel("% of Image")
            ax2.set_yticks(valid_prediction_labels)
            ax2.invert_yaxis()

        # Automatically adjust subplot params so that the subplot(s) fits into the figure area
        self.widget.figure.tight_layout(pad=3.0)

        # Explicitly set figure background color again to ensure it
        self.widget.figure.patch.set_facecolor(napari_charcoal_hex)

        self.widget.canvas.draw()

    
    def estimate_background(self):
        # Start the background painting in a new thread
        threading.Thread(target=self._paint_background_thread).start()

    def _paint_background_thread(self):
        print("Estimating background label")
        embedding_data = self.feature_data_tomotwin[:]

        # Compute the median of the embeddings
        median_embedding = np.median(embedding_data, axis=(0, 1, 2))

        # Compute the Euclidean distance from the median for each embedding
        distances = np.sqrt(
            np.sum((embedding_data - median_embedding) ** 2, axis=-1)
        )

        # Define a threshold for background detection
        # TODO note this is hardcoded
        threshold = np.percentile(distances.flatten(), 1)

        # Identify background pixels (where distance is less than the threshold)
        background_mask = distances < threshold
        indices = np.where(background_mask)

        print(
            f"Distance distribution: min {np.min(distances)} max {np.max(distances)} mean {np.mean(distances)} median {np.median(distances)} threshold {threshold}"
        )

        print(f"Labeling {np.sum(background_mask)} pixels as background")

        # TODO: optimize this because it is wicked slow
        #       once that is done the threshold can be increased
        # Update the painting data with the background class (1)
        for i in range(len(indices[0])):
            self.painting_data[indices[0][i], indices[1][i], indices[2][i]] = 1

        # Refresh the painting layer to show the updated background
        self.get_painting_layer().refresh()

    def create_embedding_plot(self):
        self.widget.embedding_figure.clear()

        # Flatten the feature data and labels to match shapes
        features = self.feature_data_tomotwin[:].reshape(-1, self.feature_data_tomotwin.shape[-1])
        labels = self.painting_data[:].flatten()

        # Filter out entries where the label is 0
        filtered_features = features[labels > 0]
        filtered_labels = labels[labels > 0]

        # Check if there are enough samples to proceed
        if filtered_features.shape[0] < 2:
            print("Not enough labeled data to create an embedding plot. Need at least 2 samples.")
            return

        # Proceed with PLSRegression as there's enough data
        self.pls = PLSRegression(n_components=2)
        self.pls_embedding = self.pls.fit_transform(filtered_features, filtered_labels)[0]

        # Original image coordinates
        z_dim, y_dim, x_dim, _ = self.feature_data_tomotwin.shape
        X, Y, Z = np.meshgrid(np.arange(x_dim), np.arange(y_dim), np.arange(z_dim), indexing='ij')
        original_coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        # Filter coordinates using the same mask applied to the features
        self.filtered_coords = original_coords[labels > 0]
        
        
        class_color_mapping = {
            label: "#{:02x}{:02x}{:02x}".format(
                int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
            ) for label, rgba in self.get_labels_colormap().items()
        }

        # Convert filtered_labels to a list of colors for each point
        point_colors = [class_color_mapping[label] for label in filtered_labels]

        # Custom style adjustments for dark theme
        napari_charcoal_hex = "#262930"
        plt.style.use('dark_background')
        self.widget.embedding_figure.patch.set_facecolor(napari_charcoal_hex)

        ax = self.widget.embedding_figure.add_subplot(111, facecolor=napari_charcoal_hex)
        scatter = ax.scatter(self.pls_embedding[:, 0], self.pls_embedding[:, 1], s=0.1, c=point_colors, alpha=1.0)

        plt.setp(ax, xticks=[], yticks=[])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')

        plt.title('PLS-DA Embedding Using Labels from Painting Layer', color='white')

        def onclick(event):
            if event.inaxes == ax:
                clicked_embedding = np.array([event.xdata, event.ydata])
                distances = np.sqrt(np.sum((self.pls_embedding - clicked_embedding) ** 2, axis=1))
                nearest_point_index = np.argmin(distances)
                nearest_image_coordinates = self.filtered_coords[nearest_point_index]
                print(f"Clicked embedding coordinates: ({event.xdata}, {event.ydata}), Image space coordinate: {nearest_image_coordinates}")

        def onselect(verts):
            path = Path(verts)
            self.update_painting_layer(path)

        # Create the LassoSelector
        self.lasso = LassoSelector(ax, onselect, useblit=True)

        cid = self.widget.embedding_canvas.mpl_connect('button_press_event', onclick)
        self.widget.embedding_canvas.draw()

    def update_painting_layer(self, path):
        # Fetch the currently active label from the painting layer
        target_label = self.get_painting_layer().selected_label
        # Start a new thread to update the painting layer with the current target label
        update_thread = Thread(target=self.paint_thread, args=(path, target_label,))
        update_thread.start()

    def paint_thread(self, lasso_path, target_label):
        # Ensure we're working with the full feature dataset
        all_features_flat = self.feature_data_tomotwin[:].reshape(-1, self.feature_data_tomotwin.shape[-1])

        # Use the PLS model to project these features into the embedding space
        all_embeddings = self.pls.transform(all_features_flat)

        # Determine which points fall within the lasso path
        contained = np.array([lasso_path.contains_point(point) for point in all_embeddings[:, :2]])

        # The shape of the original image data, to map flat indices back to spatial coordinates
        shape = self.feature_data_tomotwin.shape[:-1]

        # Iterate over all points to update the painting data where contained is True
        for idx in np.where(contained)[0]:
            # Map flat index back to spatial coordinates
            z, y, x = np.unravel_index(idx, shape)
            # Update the painting data
            self.painting_data[z, y, x] = target_label

        print(f"Painted {np.sum(contained)} pixels with label {target_label}")
        
        
class CryoCanvasWidget(QWidget):
    def __init__(self, app, parent=None):
        super(CryoCanvasWidget, self).__init__(parent)
        self.app = app
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Dropdown for selecting the model
        model_label = QLabel("Select Model")
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["Random Forest", "XGBoost"])
        model_layout = QHBoxLayout()
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_dropdown)
        layout.addLayout(model_layout)

        # Boolean options for features
        self.basic_checkbox = QCheckBox("Basic")
        self.basic_checkbox.setChecked(True)
        self.embedding_checkbox = QCheckBox("Embedding")
        self.embedding_checkbox.setChecked(True)

        features_group = QGroupBox("Features")
        features_layout = QVBoxLayout()
        features_layout.addWidget(self.basic_checkbox)
        features_layout.addWidget(self.embedding_checkbox)
        features_group.setLayout(features_layout)
        layout.addWidget(features_group)

        # Button for estimating background
        self.estimate_background_button = QPushButton("Estimate Background")
        layout.addWidget(self.estimate_background_button)

        # Dropdown for data selection
        data_label = QLabel("Select Data for Model Fitting")
        self.data_dropdown = QComboBox()
        self.data_dropdown.addItems(
            ["Current Displayed Region", "Whole Image"]
        )
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

        # Slider for adjusting thickness
        thickness_label = QLabel("Adjust Thickness")
        self.thickness_slider = QSlider(Qt.Horizontal)
        self.thickness_slider.setMinimum(0)
        self.thickness_slider.setMaximum(50)
        self.thickness_slider.setValue(10)
        thickness_layout = QHBoxLayout()
        thickness_layout.addWidget(thickness_label)
        thickness_layout.addWidget(self.thickness_slider)
        layout.addLayout(thickness_layout)

        # Connect the slider to a method to update thickness
        self.thickness_slider.valueChanged.connect(self.on_thickness_changed)
        
        # Add class distribution plot
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.embedding_figure = Figure()
        self.embedding_canvas = FigureCanvas(self.embedding_figure)
        layout.addWidget(self.embedding_canvas)

        self.setLayout(layout)

    def on_thickness_changed(self, value):
        # This method will be called whenever the slider value changes.
        # Emit a signal or directly call a method in CryoCanvasApp to update the viewer thickness
        self.app.viewer.dims.thickness = (value, ) * self.app.viewer.dims.ndim


# Initialize your application
if __name__ == "__main__":
    zarr_path = "/Users/kharrington/Data/CryoCanvas/cryocanvas_crop_006.zarr"
    app = CryoCanvasApp(zarr_path)
    # napari.run()
