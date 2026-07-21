# Laser Cutter Error Analysis

Circle center detection for laser cutter calibration and error analysis.

## Overview

This tool processes images of calibration circles arranged in a grid pattern and extracts their centers. The resulting coordinates are saved as a tensor for analyzing laser cutter accuracy and calibration errors.

## How It Works

1. **Images** are named `circle_NxM.png` where N = row, M = column
2. **Detection** uses OpenCV contour detection to find circle centers
3. **Output** is a 2×N×M tensor where:
   - `[0, row, col]` = x-coordinate
   - `[1, row, col]` = y-coordinate

## Project Structure
laser_cutter_accuracy/
├── contour_detection.py # Circle center detection
├── tensor_output.py # Grid processing & tensor creation
├── images_circle_nxm/ # Input images (place your images here)
│ ├── circle_1x1.png
│ ├── circle_2x1.png
│ └── ...
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore file
└── README.md # This file


## Installation

git clone https://github.com/aiturner/laser-cutter-error-analysis.git
cd laser-cutter-error-analysis

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt

## Requirements

Python 3.8+
OpenCV
NumPy


