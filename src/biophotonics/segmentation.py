import numpy as np
from PyQt5.QtWidgets import QWidget, QPushButton, QCheckBox, QVBoxLayout, QLabel

from biophotonics.utils import TextBoxWidget
from sklearn.exceptions import NotFittedError

class RFWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

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

        if len(raw_img.shape) > 2:
            raise Exception('please ensure that a 2D image is selected; segmentation of 3D images will take a very long time')

        name = self.viewer.layers.selection.active.name

        if self.adaptive_background_checkbox.isChecked() & (raw_img[training_labels == 1].any()):
            self.mean = np.mean(raw_img[training_labels == 1])
        else:
            self.mean = 12918.499373693521

        features = self.make_features(raw_img)

        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                     max_depth=100, max_samples=0.05)

        self.clf = future.fit_segmenter(training_labels, features, clf)

        result = self.predict_segmenter(features, self.clf)
        self.viewer.add_image(result, name=f'segmentation probabilities')

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

        return features

    def predict_segmenter(self, features, clf):
        """
        taken from https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/future/trainable_segmentation.py#L89-L118
        Segmentation of images using a pretrained classifier.
        Parameters
        ----------
        features : ndarray
            Array of features, with the last dimension corresponding to the number
            of features, and the other dimensions are compatible with the shape of
            the image to segment, or a flattened image.
        clf : classifier object
            trained classifier object, exposing a ``predict`` method as in
            scikit-learn's API, for example an instance of
            ``RandomForestClassifier`` or ``LogisticRegression`` classifier. The
            classifier must be already trained, for example with
            :func:`skimage.segmentation.fit_segmenter`.
        Returns
        -------
        output : ndarray
            Labeled array, built from the prediction of the classifier.
        """
        sh = features.shape
        if features.ndim > 2:
            features = features.reshape((-1, sh[-1]))

        try:
            predicted_labels = clf.predict(features)
            print(f'{predicted_labels.shape}')
            predicted_labels = clf.predict_proba(features)
            print(f'{predicted_labels.shape}')
        except NotFittedError:
            raise NotFittedError(
                "You must train the classifier `clf` first"
                "for example with the `fit_segmenter` function."
            )
        except ValueError as err:
            if err.args and 'x must consist of vectors of length' in err.args[0]:
                raise ValueError(
                    err.args[0] + '\n' +
                    "Maybe you did not use the same type of features for training the classifier."
                )
            else:
                raise err
        if len(predicted_labels.shape) ==1:
            output = predicted_labels.reshape(sh[:-1])
        elif len(predicted_labels.shape) ==2:
            feature_dim = predicted_labels.shape[-1]
            s = list(sh[:-1]) + [feature_dim]
            output = predicted_labels.reshape(s)
            output = np.rollaxis(output, 2, 0)
        return output