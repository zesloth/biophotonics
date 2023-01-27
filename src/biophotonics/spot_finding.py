import numpy as np
from PyQt5.QtWidgets import QWidget, QPushButton, QCheckBox, QVBoxLayout, QLabel
from biophotonics.utils import TextBoxWidget, DropdownMenu
import napari


class SpotFinderWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.title = QLabel()
        self.title.setText('spot finding')

        self.first_sigma = TextBoxWidget('first gaussian sigma', '1')
        self.second_sigma = TextBoxWidget('second gaussian sigma', '10')

        self.find_spots_button = QPushButton('compute difference of gaussian')
        self.find_spots_button.clicked.connect(self.find_spots)

        self.threshold_value = TextBoxWidget('spot finder threshold', '0.001')
        self.min_spot_distance = TextBoxWidget('minimum spot distance (px)', '3')

        self.threshold_button = QPushButton('threshold')
        self.threshold_button.clicked.connect(self.threshold_dog)

        self.spot_intensity_layer = DropdownMenu('spot intensity layer')
        self.nucleus_intensity_layer = DropdownMenu('dapi max projection')
        self.segmentation_layer = DropdownMenu('segmentation')
        self.spot_location_layer = DropdownMenu('spot position layer')
        self.layer_selection_dropdowns = [self.spot_intensity_layer,
                                          self.nucleus_intensity_layer,
                                          self.segmentation_layer,
                                          self.spot_location_layer]

        self.savepath = TextBoxWidget('save spot dataframe to:', 'cy5_spots.pkl')
        self.save_spots_button = QPushButton('save spot information')
        self.save_spots_button.clicked.connect(self.save_spots)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.title)

        self.layout().addWidget(self.first_sigma)
        self.layout().addWidget(self.second_sigma)
        self.layout().addWidget(self.find_spots_button)
        self.layout().addWidget(self.threshold_value)
        self.layout().addWidget(self.min_spot_distance)
        self.layout().addWidget(self.threshold_button)

        for dropdown in self.layer_selection_dropdowns:
            self.layout().addWidget(dropdown)

        self.layout().addWidget(self.savepath)
        self.layout().addWidget(self.save_spots_button)

        self.viewer.layers.events.inserted.connect(self._on_insert)
        self.viewer.layers.events.removed.connect(self._update_comboboxes)

    def _on_insert(self, event):
        layer = event.value
        self._update_comboboxes(None)
        layer.events.name.connect(self._update_comboboxes)


    def _update_comboboxes(self, event):

        for box in self.layer_selection_dropdowns:
            box.clear()
            for l in self.viewer.layers:
                box.add(l.name)

    def threshold_dog(self):
        from skimage.feature import peak_local_max

        img = self.viewer.layers.selection.active.data
        name = self.viewer.layers.selection.active.name
        threshold = float(self.threshold_value.get_text())
        min_distance = int(self.min_spot_distance.get_text())
        coords = peak_local_max(img * (img > threshold), min_distance=min_distance)

        self.viewer.add_points(coords)

    def find_spots(self):
        from skimage.filters import gaussian
        img = self.viewer.layers.selection.active.data
        name = self.viewer.layers.selection.active.name

        sigma_0 = float(self.first_sigma.get_text())
        sigma_1 = float(self.second_sigma.get_text())

        dog = gaussian(img, sigma_0) - gaussian(img, sigma_1)
        self.viewer.add_image(dog, name=f'difference of gaussian {name}')

    def save_spots(self):
        import pandas as pd
        save_path = self.savepath.get_text()

        segmentation_name = self.segmentation_layer.get_text()
        if (segmentation_name is None) or (not isinstance(self.viewer.layers[segmentation_name], napari.layers.labels.labels.Labels)):
            raise ReferenceError('please select the correct segmentation layer')
        segmentation = self.viewer.layers[segmentation_name].data

        points = self.spot_location_layer.get_text()
        print(points)
        if (points is None) or (not isinstance(self.viewer.layers[points], napari.layers.points.points.Points)):
            raise ReferenceError('please select the correct spot position layer')
        points = self.viewer.layers[points].data.astype(int)

        spot_intensities = self.spot_intensity_layer.get_text()
        if (spot_intensities is None) or (spot_intensities not in self.viewer.layers):
            raise ReferenceError('please select a valid spot intensity layer')
        spots = self.viewer.layers[spot_intensities].data

        nucl_intensities = self.nucleus_intensity_layer.get_text()
        if (nucl_intensities is None) or (nucl_intensities not in self.viewer.layers):
            raise ReferenceError('please select a valid nucleus layer')
        nuclei = self.viewer.layers[nucl_intensities].data

        output_data = []
        for x, y in points:

            if segmentation[x, y] > 0:
                output_data.append([x, y, segmentation[x, y], spots[x,y], nuclei[x,y]])

        df = pd.DataFrame(output_data, columns=['row', 'column', 'cell_id', 'spot_intensities', 'nuclear_intensity'])

        df.to_pickle(save_path)