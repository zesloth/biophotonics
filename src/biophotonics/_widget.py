"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QVBoxLayout

from biophotonics.segmentation import RFWidget
from biophotonics.watershed import WatershedWidget

if TYPE_CHECKING:
    import napari


class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.rf_widget = RFWidget(self.viewer)
        self.ws_widget = WatershedWidget(self)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.rf_widget)
        self.layout().addWidget(self.ws_widget)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")
