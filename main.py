import logging
from typing import Any, Dict

import cv2
import numpy as np

from modules.utils import get_arguments
from modules.video_stream import VideoStream
from modules.yolo_model import YOLOModel


class MainApp:
    """
    Main application for video streaming, YOLO object detection,
    and display logic integration.
    """

    def __init__(self, args: Dict[str, str]) -> None:
        """
        Initialize the MainApp with model, video stream, and output settings.

        Args:
            args (Dict[str, str]): Command-line arguments with keys:
                - "model": Path to the YOLO model file.
                - "device": Device to run inference on ("cpu", "cuda", "mps").
                - "input": Path to video file or None for camera.
                - "show": Whether to display the video.
        """
        self.model = YOLOModel(args["model"], args["device"])
        self.video_stream = VideoStream(args["input"])
        self.show = args.get("show")

    @staticmethod
    def check_targets(detections: Any) -> bool:
        """
        Determines whether there is at least one detection whose class ID is in
        `target_class_ids`

        Args:
            detections (Iterable[Any]): The output from YOLOModel.process_frame()

        Returns:
            bool: True if at least one detection matches one of the target classes
                and meets or exceeds the threshold; False otherwise.
        """
        for det in detections:
            class_ids = det.boxes.cls.cpu().numpy().astype(int)
            confidences = det.boxes.conf.cpu().numpy()
            for cls_id, conf in zip(class_ids, confidences):
                if cls_id in [2, 3, 7] and conf >= 0.5:
                    return True
        return False

    def _cleanup(self) -> None:
        """
        Release resources and close all OpenCV windows.
        """
        self.video_stream.release()
        if self.show:
            cv2.destroyAllWindows()

    def run(self) -> None:
        """
        Run the main loop: read frames, detect objects, annotate, display, and save.
        """
        while True:
            grabbed, frame = self.video_stream.read_frame()
            if not grabbed or frame is None:
                logging.info("No more frames or failed to capture.")
                break

            # Define polygon vertices for the region of interest (ROI)
            pts = np.array([
                [500, 1200],
                [500, 800],
                [1050, 500],
                [1630, 500],
                [1550, 1200],
            ], dtype=np.int32)

            # Compute bounding rectangle: (x, y) coords of top-left corner, plus width and height
            x, y, w_box, h_box = cv2.boundingRect(pts)

            # Create a black mask and fill the polygon area with white
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], color=(255,))

            # Apply mask to keep only the ROI in the frame
            masked = cv2.bitwise_and(frame, frame, mask=mask)

            # Draw the ROI contour on the original frame
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=4)

            # Crop the masked ROI and run detection on it
            roi = masked[y: y + h_box, x: x + w_box]
            results = self.model.process_frame(roi)
            has_target = self.check_targets(results)
            logging.info(f"Detected target in ROI: {has_target}")

            if self.show:
                cv2.imshow("Car & Motorcycle Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logging.info("Exit key pressed.")
                    break

        self._cleanup()


def main():
    """
    Entry point for the application. Parses arguments, initializes the app, and runs it.
    """
    args = get_arguments()
    app = MainApp(args)
    app.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
    )
    main()
