from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QLineEdit


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
