from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel
import numpy as np
from biophotonics.utils import TextBoxWidget
import napari
class WatershedWidget(QWidget):
    def __init__(self, parent):
        super().__init__()

        self.parent = parent
        self.viewer = parent.viewer

        #widgets
        self.title = QLabel()
        self.title.setText('biophotonics')

        self.seed_intensity_threshold = TextBoxWidget('minimum seed intensity', '15000')
        self.min_seed_distance = TextBoxWidget('minimum seed distance', '20')

        self.generate_seeds_button = QPushButton("generate seeds")
        self.generate_seeds_button.clicked.connect(self.generate_seeds)
        self.watershed_button = QPushButton("watershed")
        self.watershed_button.clicked.connect(self.watershed)

        # attributes

        # layout
        self.setLayout(QVBoxLayout())

        self.layout().addWidget(self.seed_intensity_threshold)
        self.layout().addWidget(self.min_seed_distance)
        self.layout().addWidget(self.generate_seeds_button)
        self.layout().addWidget(self.watershed_button)


    def generate_seeds(self):
        from skimage.feature import peak_local_max
        from skimage.filters import gaussian

        img = self.viewer.layers.selection.active.data
        img = np.max(img[15:25, ...], axis=0)


        name = self.viewer.layers.selection.active.name

        thresh = int(self.seed_intensity_threshold.get_text())
        dist = int(self.min_seed_distance.get_text())

        img_mask = img>thresh
        coords = peak_local_max(gaussian(img_mask, 1), min_distance=dist)

        self.viewer.add_points(coords, name=f'seeds')


    def watershed(self):
        import scipy.ndimage as ndi
        from skimage.segmentation import watershed
        from skimage.filters import gaussian
        img = self.viewer.layers.selection.active.data
        print(self.viewer.layers.selection.active)

        seeds = self.viewer.layers['seeds'].data

        # ensure 2d seeds for the moment
        if seeds.shape[-1] > 2:
            seeds = seeds[:,1:]

        landscape = img[0,...] - 2*img[1,...] - img[2,...]

        borders = landscape < 1

        landscape = gaussian(1.*landscape, 2)
        self.viewer.add_image(landscape, name='gaussian')

        seed_map = np.zeros_like(landscape, dtype=bool)
        seed_map[tuple(seeds.T.astype(int))] = 1
        seed_map, _ = ndi.label(seed_map)

        segmentation = watershed(landscape, seed_map, mask=borders)
        # # dummy for multi-image datasets:
        segmentation = segmentation[np.newaxis, ...]
        segmentation = np.concatenate(3*[segmentation], axis=0)
        self.viewer.add_labels(segmentation, name='segmentation')