# Laser Cutter Calibration

Analyses images taken of a chessboard calibration grid of crosses and modifies galvo input to reduce error.

## Project Structure

```
laser_cutter_accuracy/
|
├── error_tensor/
│   ├── main.py
│   └── center_coords_linefitting.py
│   └── tensor_from_directory.py
│   └── pix_to_mm.py
|
├── modify_inputs/
|
│   ├── main.py
│   └── convert_polynomial_fit.py
│   └── csv-mod.py      
|      
├── plot_data/
│   ├── plot_2.py
│   └── plot_raw_data.py   
|           
├── caibration_images/
|
├── I_O_data/
|
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```
## Requirements

```bash
opencv-python
numpy
pandas
matplotlib
```
## Usage

Take high resolution images, name them n (m) were n is the row and m is column.
Put these in a directory e.g caibration_images
Run error_tensor/main.py

```bash
python3 error_tensor/main.py
```
When prompted enter the size of the calibration grid and position of the central point.
Decide if you want to use maunal center detection or automatic detection.
Enter the directory path containing the calibration images.

Convert the created tensor from pixels to millimeters then save the output
(saved as offset_data.npy)

Next run the calibration and input modification
Run modify_inputs/main.py

```bash
python3 modify_inputs/main.py
```
When prompted enter txt file containing commanded galvo coordinates.
The function will return the modified coordinates after the transformation has been applied.

Inorder to see the original points against the new points use the plotting function
```bash
python3 plotting_data/plot_2.py
```


