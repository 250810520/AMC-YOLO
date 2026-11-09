"""Evaluate YOLO11 with AIFI, CARAFE, and MCA."""

from text import main


if __name__ == "__main__":
    main("AIFI-CARAFE-MCA-best.pt", "amc_result", plots=False)
