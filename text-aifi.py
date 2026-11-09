"""Evaluate YOLO11 with AIFI."""

from text import main


if __name__ == "__main__":
    main("best-AIFI.pt", "aifi_result", plots=False)
