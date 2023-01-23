import numpy as np
from PyQt5.QtWidgets import QWidget, QPushButton, QCheckBox, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout


class TextBoxWidget(QWidget):
    def __init__(self, box_name, default_text):
        super().__init__()

        self.setLayout(QHBoxLayout())

        self.root_label = QLabel(f'{box_name}: ')

        self.root_textbox = QLineEdit(self)
        self.root_textbox.setText(default_text)

        self.layout().addWidget(self.root_label)
        self.layout().addWidget(self.root_textbox)

    def get_text(self):
        return self.root_textbox.text()


class RFWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        # load dummy image

        self.title = QLabel()
        self.title.setText('biophotonics')

        self.dimension_input_first = TextBoxWidget('z level start', '15')
        self.dimension_input_last = TextBoxWidget('z level finish', '25')

        self.max_projection_button = QPushButton("max_projection")
        self.max_projection_button.clicked.connect(self.max_projection)

        self.random_forest_button = QPushButton("classify selected layer")
        self.random_forest_button.clicked.connect(self.train_classifier)

        self.adaptive_background_checkbox = QCheckBox('adapt background')
        self.adaptive_background_checkbox.setChecked(True)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.title)

        self.layout().addWidget(self.dimension_input_first)
        self.layout().addWidget(self.dimension_input_last)
        self.layout().addWidget(self.max_projection_button)

        self.layout().addWidget(self.random_forest_button)
        self.layout().addWidget(self.adaptive_background_checkbox)

    def train_classifier(self):
        from skimage import future
        from sklearn.ensemble import RandomForestClassifier

        if 'Labels' in self.viewer.layers:
            training_labels = self.viewer.layers['Labels'].data
        else:
            raise Exception('training labels must be in a layer called "Labels"')

        raw_img = self.viewer.layers.selection.active.data
        name = self.viewer.layers.selection.active.name

        if self.adaptive_background_checkbox.isChecked() & (raw_img[training_labels == 1].any()):
            self.mean = np.mean(raw_img[training_labels == 1])
        else:
            self.mean = 12918.499373693521

        features = self.make_features(raw_img)

        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                     max_depth=100, max_samples=0.05)

        self.clf = future.fit_segmenter(training_labels, features, clf)

        result = future.predict_segmenter(features, self.clf)
        self.viewer.add_labels(result, name=f'{name} segmentation')
        return


    def max_projection(self):
        z0 = self.dimension_input_first.get_text()
        z1 = self.dimension_input_last.get_text()
        z0, z1 = int(z0), int(z1)

        img, name = self.viewer.layers.selection.active.data, self.viewer.layers.selection.active.name
        max_proj = np.max(img[z0:z1, ...], axis=0)
        self.viewer.add_image(max_proj, name=f'{name} max proj')
        self.viewer.add_labels(np.zeros_like(max_proj), name=f'Labels')


    def make_features(self, img):
        features = []
        features.append(self.make_simple_features(img))

        if len(features) == 0:
            raise ValueError('No features selected!')

        if len(features) == 1:
            return features[0]
        else:
            return np.concatenate(features, axis=-1)

    def make_simple_features(self, img):
        from skimage import feature
        from skimage.filters import gaussian, sobel
        from functools import partial
        features_func = partial(feature.multiscale_basic_features,
                                intensity=True, edges=False, texture=True, sigma_min=1, sigma_max=16)
        features = features_func(img)

        gaussians = []
        for sigma in [1, 10]:
            gaussians.append(gaussian(img, sigma)[...,np.newaxis])

        sob = sobel(img)[..., np.newaxis]
        features = np.concatenate([features, sob, gaussians[1]-gaussians[0]], axis=-1)

        self.viewer.add_image(features, name='simple_features')

        return features
