"""
This module is an example of a barebones QWidget plugin for napari
It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

This file provides an interface for functionality that will be used in the practical.
Look into the respective .py files for hte actual code.

Code is organised as three separate widgets, because putting all functions into one large
widget runs into problems on small displays.
"""
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QVBoxLayout

from biophotonics.segmentation import RFWidget
from biophotonics.watershed import WatershedWidget
from biophotonics.spot_finding import SpotFinderWidget

if TYPE_CHECKING:
    import napari


class RandomForest(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.rf_widget = RFWidget(self.viewer)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.rf_widget)


class Watershed(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.ws_widget = WatershedWidget(self)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.ws_widget)


class SpotFinder(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.sf_widget = SpotFinderWidget(self.viewer)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.sf_widget)
