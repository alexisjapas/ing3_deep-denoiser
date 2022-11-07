from qtpy.QtWidgets import QWidget, QVBoxLayout
from magicgui import magicgui
from napari.qt.threading import thread_worker
from napari.layers import Image
import torch
import numpy as np
from tkinter.filedialog import askdirectory

import noises
from inferences import infer


class AppDenoiser(QWidget):
    def __init__(self, viewer) -> None:
        super().__init__()
        self.viewer = viewer

    def update_viewer(self, layer):
        layer_name = f"{layer['name']}_{layer['type']}"
        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = layer["data"]
        else:
            self.viewer.add_image(layer["data"], name=layer_name)

    def noise(self, image: Image, sap_density=0.20):
        @thread_worker(connect={"returned": self.update_viewer})
        def _noise():
            return {"name": image.name, "data": noises.salt_and_pepper(np.array(image.data), sap_density), "type": "noisy"}
        _noise()

    def lin_denoise(self, image: Image):
        @thread_worker(connect={"returned": self.update_viewer})
        def _lin_denoise():
            pass
        _lin_denoise()

    def deep_denoise(self, image: Image, model_path="model.pth", device="cpu"):
        @thread_worker(connect={"returned": self.update_viewer})
        def _deep_denoise():
            # Setup model
            model = torch.load(model_path).to(device)
            return {"name": image.name, "data": infer(np.array(image.data), model, device), "type": "corr"}
        _deep_denoise()


if __name__ == "__main__":
    import napari

    # Create a viewer
    viewer = napari.Viewer()
    Denoiser = AppDenoiser(viewer)

    # Adds noiser
    noiser = magicgui(Denoiser.noise, call_button="noise")
    viewer.window.add_dock_widget(noiser, name="Noise functions")
    # Adds lin denoiser
    lin_denoiser = magicgui(Denoiser.lin_denoise, call_button="Denoise with linear filters")
    #viewer.window.add_dock_widget(lin_denoiser, name="Linear denoiser")
    # Adds deep denoiser
    deep_denoiser = magicgui(Denoiser.deep_denoise, call_button="Denoise with deep learning", device={"choices": ["cpu", "cuda"]})
    viewer.window.add_dock_widget(deep_denoiser, name="Deep denoiser")

    # Run app
    napari.run()
