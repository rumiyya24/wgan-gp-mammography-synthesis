import sys
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from PyQt6.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QWidget,
    QHBoxLayout, QLineEdit, QMessageBox, QGridLayout, QScrollArea, QComboBox
)
from PyQt6.QtGui import QPixmap, QImage, QCursor, QColor, QPainter
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PIL import Image
import os
import pandas as pd


# ---------------- GAN Generator ----------------
class Generator(Model):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.height, self.width, self.channels = img_shape

        def block(units, normalize=True):
            layers_block = [layers.Dense(units)]
            if normalize:
                layers_block.append(layers.BatchNormalization(momentum=0.8))
            layers_block.append(layers.LeakyReLU(0.2))
            return layers_block

        self.model_fc = tf.keras.Sequential(
            block(128, normalize=False) +
            block(256) +
            block(512) +
            block(1024) +
            [layers.Dense(int(np.prod(img_shape)), activation="tanh")]
        )

        self.conv_refine = tf.keras.Sequential([
            layers.Conv2D(32, kernel_size=3, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Conv2D(32, kernel_size=3, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Conv2D(self.channels, kernel_size=3, padding="same", activation="tanh"),
        ])

    def call(self, z, training=False):
        x = self.model_fc(z)
        x = tf.reshape(x, (-1, self.height, self.width, self.channels))
        return self.conv_refine(x, training=training)


def load_gan_generator():
    latent_dim = 100
    img_shape = (256, 256, 1)
    generator = Generator(latent_dim, img_shape)
    _ = generator(tf.random.normal((1, latent_dim)))  # Build model

    try:
        generator.load_weights("generator_epoch_500.weights.h5")
        print("Generator weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load weights: {e}")

    return generator


REAL_IMAGES_FOLDER = "/Users/rumiyyaalili/Downloads/InBreast_Aligned_Images"


def generate_fake_image(model):
    z = np.random.normal(0, 1, (1, 100)).astype(np.float32)
    fake_img = model.predict(z, verbose=0).squeeze()
    fake_img = ((fake_img + 1) * 127.5).astype(np.uint8)
    return Image.fromarray(fake_img, mode="L")


def load_real_image():
    try:
        img_name = random.choice(os.listdir(REAL_IMAGES_FOLDER))
        img_path = os.path.join(REAL_IMAGES_FOLDER, img_name)
        return Image.open(img_path).convert("L")
    except Exception as e:
        print(f"Error loading real image: {e}")
        return Image.new("L", (256, 256))


# ---------------- GUI Classes ----------------

class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()

    def enterEvent(self, event):
        self.setStyleSheet("border: 2px solid #888;")

    def leaveEvent(self, event):
        self.setStyleSheet("border: none;")


class UserInput(QWidget):
    def __init__(self, start_callback):
        super().__init__()
        self.setWindowTitle("User Input")
        self.setGeometry(300, 300, 400, 200)
        self.start_callback = start_callback
        self.layout = QVBoxLayout()

        self.name_label = QLabel("Enter your name:")
        self.name_input = QLineEdit()

        self.role_label = QLabel("Select your role:")
        self.role_dropdown = QComboBox()
        self.role_dropdown.addItems([
            "Breast Surgeon",
            "Doctor-mammolog",
            "Radiologist-mammolog",
            "Radiologist-other",
            "Doctor-other",
            "Intern medical student",
            "Senior medical student"
        ])

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_app)

        self.layout.addWidget(self.name_label)
        self.layout.addWidget(self.name_input)
        self.layout.addWidget(self.role_label)
        self.layout.addWidget(self.role_dropdown)
        self.layout.addWidget(self.start_button)

        self.setLayout(self.layout)

    def start_app(self):
        name = self.name_input.text().strip()
        role = self.role_dropdown.currentText()
        if name and role:
            self.start_callback(name, role)
            self.close()
        else:
            QMessageBox.warning(self, "Input Error", "Please enter both name and role")


class ImageReviewApp(QWidget):
    def __init__(self, user_name, user_role):
        super().__init__()
        self.user_name = user_name
        self.user_role = user_role  # Save the role
        self.setWindowTitle("GAN Image Test")
        self.generator = load_gan_generator()
        self.user_choices = []
        self.start_time = None
        self.change_count = 0
        self.selected_answer = None

        # Image display setup
        self.image_labels = [QLabel() for _ in range(3)]
        self.image_clickable_labels = []
        for i, label in enumerate(self.image_labels):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setSizePolicy(label.sizePolicy().horizontalPolicy(), label.sizePolicy().verticalPolicy())
            label.setStyleSheet("cursor: pointer;")  # Add a pointer cursor on hover
            label.mousePressEvent = lambda event, idx=i: self.select_image(idx)  # Handle click on images

        # Buttons
        self.submit_button = QPushButton("Submit")
        self.next_button = QPushButton("Next")
        self.complete_button = QPushButton("Complete")

        self.submit_button.clicked.connect(self.submit_choice)
        self.next_button.clicked.connect(self.load_next_case)
        self.complete_button.clicked.connect(self.save_results)

        # Layout setup
        self.scroll = QScrollArea()
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)

        for i in range(3):
            self.grid_layout.addWidget(self.image_labels[i], 0, i)

        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.grid_widget)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.scroll)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.submit_button)
        btn_layout.addWidget(self.next_button)
        btn_layout.addWidget(self.complete_button)

        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

        self.images = []
        self.load_next_case()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image_sizes()

    def update_image_sizes(self):
        width = self.scroll.width() // 3 - 40
        height = width
        for i, img in enumerate(self.images):
            if img:
                resized = img.resize((width, height))
                img_array = np.asarray(resized)
                qimg = QImage(
                    img_array.data, img_array.shape[1], img_array.shape[0],
                    img_array.strides[0], QImage.Format.Format_Grayscale8
                )
                self.image_labels[i].setPixmap(QPixmap.fromImage(qimg))

    def select_image(self, index):
        if self.selected_answer is None:
            self.selected_answer = index
        elif self.selected_answer != index:
            self.change_count += 1
            self.selected_answer = index

        # Update the image sizes dynamically when clicked
        self.update_selected_image_sizes()

    def update_selected_image_sizes(self):
        # Increase size of the selected image and revert others to normal size
        for i, img in enumerate(self.images):
            width = self.scroll.width() // 3 - 40
            height = width
            if i == self.selected_answer:
                resized = img.resize((int(width * 1.2), int(height * 1.2)))  # Make selected image bigger
            else:
                resized = img.resize((width, height))  # Normal size for unselected images

            img_array = np.asarray(resized)
            qimg = QImage(
                img_array.data, img_array.shape[1], img_array.shape[0],
                img_array.strides[0], QImage.Format.Format_Grayscale8
            )
            self.image_labels[i].setPixmap(QPixmap.fromImage(qimg))

    def load_next_case(self):
        self.start_time = time.time()
        self.change_count = 0
        self.selected_answer = None
        fake_img = generate_fake_image(self.generator)
        self.images = [load_real_image() for _ in range(2)] + [fake_img]
        random.shuffle(self.images)
        self.correct_index = self.images.index(fake_img)
        self.update_image_sizes()

        for i in range(3):
            self.image_labels[i].setStyleSheet("")  # Reset styles
        self.submit_button.setDisabled(False)
        self.next_button.setDisabled(True)
        self.complete_button.setDisabled(True)

    def submit_choice(self):
        if self.selected_answer is None:
            return
        elapsed = round(time.time() - self.start_time, 2)
        correct = self.selected_answer == self.correct_index
        self.user_choices.append((correct, elapsed, self.change_count))
        self.image_labels[self.selected_answer].setStyleSheet("border: 3px solid red;")
        self.image_labels[self.correct_index].setStyleSheet("border: 3px solid green;")
        self.submit_button.setDisabled(True)
        self.complete_button.setDisabled(False)
        self.next_button.setDisabled(False)

    def save_results(self):
        filename = "user_results.csv"
        df = pd.DataFrame(self.user_choices, columns=["Correct", "Response Time (s)", "Changed Mind Count"])
        df["Name"] = self.user_name
        df["Role"] = self.user_role  # Add the role column

        if not os.path.exists(filename):
            df.to_csv(filename, index=False)
        else:
            df.to_csv(filename, mode="a", header=False, index=False)

        QMessageBox.information(self, "Results Saved", f"Results saved to {filename}")
        self.close()


# ---------------- Entry Point ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)


    def start_main_app(name, role):
        global main_window_instance
        main_window_instance = ImageReviewApp(name, role)
        main_window_instance.show()


    input_window = UserInput(start_main_app)
    input_window.show()
    sys.exit(app.exec())