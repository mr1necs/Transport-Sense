from argparse import ArgumentParser


def get_arguments() -> dict[str, str]:
    """
    Parse command-line arguments for the application.

    Returns:
        dict[str, str]: Dictionary of parsed command-line arguments.
    """
    parser = ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Path to input video (e.g. cvtest.avi)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Path to load YOLO model (or YOLO model name (e.g. yolo11n.pt))",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device to use: 'mps', 'cuda', or 'cpu'",
    )
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        default=False,
        help="Show video on screen (default: False)",
    )
    return vars(parser.parse_args())
