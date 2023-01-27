from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QLineEdit
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QComboBox


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


class DropdownMenu(QWidget):
    def __init__(self, menu_name, *args):
        super().__init__()

        self.setLayout(QHBoxLayout())

        self.root_label = QLabel(f'{menu_name}: ')

        self.content = []
        self.root_dropdown = QComboBox(self)
        for a in args:
            self.add(a)

        self.layout().addWidget(self.root_label)
        self.layout().addWidget(self.root_dropdown)

    def add(self, a):
        b = str(a)
        if not b in self.content:
            self.root_dropdown.addItem(b)
            self.content.append(b)

    def clear(self):
        self.content = []
        self.root_dropdown.clear()

    def get_text(self):
        return self.root_dropdown.currentText()
