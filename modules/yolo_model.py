import logging

from torch import cuda
from torch.backends import mps
from ultralytics import YOLO


class YOLOModel:
    """
    Encapsulates the YOLO model for device selection, loading, and object detection.
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        """
        Initialize the YOLOModel.

        Args:
            model_path (str): Path to the YOLO model file.
            device (str): Preferred device ("cpu", "cuda", "mps").
        """
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        self.device = self._select_device(device)
        self.model = self._load_model(model_path)
        self.confidence_threshold = 0.5

    def _select_device(self, preferred_device: str = "cpu") -> str:
        """
        Select the computation device.

        Args:
            preferred_device (str): Preferred device ("cpu", "cuda", "mps").

        Returns:
            str: The available device as a string.
        """
        if preferred_device == "mps" and mps.is_available():
            device = "mps"
        elif preferred_device == "cuda" and cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        if device != preferred_device:
            logging.warning(
                f"Preferred device '{preferred_device}' is not available. "
                f"Using '{device}' instead."
            )

        logging.info(f"Selected device: {device}")
        return device

    def _load_model(self, model_path: str) -> YOLO:
        """
        Load the YOLO model onto the selected device.

        Args:
            model_path (str): Path to the YOLO model file.

        Returns:
            YOLO: The loaded YOLO model.
        """
        try:
            model = YOLO(model_path).to(self.device)
            logging.info("Model loaded successfully.")
            return model
        except Exception as exc:
            logging.error(f"Error loading YOLO model: {exc}")
            raise

    def process_frame(self, frame) -> list:
        """
        Perform object detection on a frame.

        Args:
            frame: Input frame (image).

        Returns:
            list: List of detected objects (YOLO results for classes).
        """

        return self.model(frame)
