from qtpy.QtWidgets import QWidget, QVBoxLayout
from magicgui import magicgui
from napari.qt.threading import thread_worker


class Denoiser(QWidget):
    def __init__(self, viewer) -> None:
        super().__init__()
        self.viewer = viewer
        # Defining layout
        layout = QVBoxLayout()
        layout.addWidget(self.denoise.native)
        self.setLayout(layout)

    @magicgui(call_button="Denoise")
    def denoise(self, n_layers=18):
        def _update_viewer(corrected_image):
            if "corrected_image" in self.viewer.layers:
                self.viewer.layers["corrected_image"] = self.corrected_image
            else:
                self.viewer.add_image(self.corrected_image, name=corrected_image)

        @thread_worker
        def _denoise():
            return corrected_image

        # Denoise
        worker = _denoise()
        worker.returned.connect(_update_viewer)
        worker.start()


if __name__ == "__main__":
    import napari

    # Create a viewer
    viewer = napari.Viewer()

    # Adds widget to the viewer
    cleaner = Denoiser(viewer)
    viewer.window.add_dock_widget(cleaner, name="Deep cleaner")

    # Run app
    napari.run()
