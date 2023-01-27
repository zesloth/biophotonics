from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel
import numpy as np
import pandas as pd
from biophotonics.utils import TextBoxWidget
import napari


class WatershedWidget(QWidget):
    def __init__(self, parent):
        super().__init__()

        self.parent = parent
        self.viewer = parent.viewer

        # widgets
        self.title = QLabel()
        self.title.setText('biophotonics')

        self.seed_intensity_threshold = TextBoxWidget('minimum nucleus intensity', '15000')
        self.min_seed_distance = TextBoxWidget('minimum nucleus distance', '20')

        self.generate_seeds_button = QPushButton("find nuclei")
        self.generate_seeds_button.clicked.connect(self.generate_seeds)
        self.watershed_button = QPushButton("watershed")
        self.watershed_button.clicked.connect(self.watershed)

        self.watershed_save_path = TextBoxWidget('segmentation save path', 'segmentation.tif')
        self.watershed_save_button = QPushButton("save segmentation")
        self.watershed_save_button.clicked.connect(self.save_watershed)

        # layout
        self.setLayout(QVBoxLayout())

        self.layout().addWidget(self.seed_intensity_threshold)
        self.layout().addWidget(self.min_seed_distance)
        self.layout().addWidget(self.generate_seeds_button)
        self.layout().addWidget(self.watershed_button)
        self.layout().addWidget(self.watershed_save_path)
        self.layout().addWidget(self.watershed_save_button)

    def generate_seeds(self):
        from skimage.feature import peak_local_max
        from skimage.filters import gaussian

        img = self.viewer.layers.selection.active.data
        img = np.max(img[15:25, ...], axis=0)

        thresh = int(self.seed_intensity_threshold.get_text())
        dist = int(self.min_seed_distance.get_text())

        img_mask = img > thresh
        coords = peak_local_max(gaussian(img_mask, 1), min_distance=dist)

        self.viewer.add_points(coords, name=f'nuclei')

    def watershed(self):
        import scipy.ndimage as ndi
        from skimage.segmentation import watershed
        from skimage.filters import gaussian
        img = self.viewer.layers.selection.active.data
        print(self.viewer.layers.selection.active)

        seeds = self.viewer.layers['nuclei'].data

        # ensure 2d seeds for the moment
        if seeds.shape[-1] > 2:
            seeds = seeds[:, 1:]

        landscape = img[0, ...] - 2 * img[1, ...] - img[2, ...]

        borders = landscape < 1

        landscape = gaussian(1. * landscape, 2)

        seed_map = np.zeros_like(landscape, dtype=bool)
        seed_map[tuple(seeds.T.astype(int))] = 1
        seed_map, _ = ndi.label(seed_map)

        segmentation = watershed(landscape, seed_map, mask=borders)
        self.viewer.add_labels(segmentation, name='segmentation')

    def save_watershed(self):
        from skimage import io
        if not 'segmentation' in self.viewer.layers:
            raise ReferenceError('please ensure there is a layer named "segmentation"')
        save_path = self.watershed_save_path.get_text()
        img = self.viewer.layers['segmentation'].data
        io.imsave(save_path, img)

        labels = set(img.ravel()) - {0}

        pd.DataFrame({'cell_label':list(labels)}).to_pickle(f'{save_path.split(".")[0]}.pkl')

