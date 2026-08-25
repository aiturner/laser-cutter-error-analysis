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
## Methodology
Part 1- Measurement

This report documents the methodology for detecting cross centres in images for laser cutter calibration and accuracy analysis. The system processes images of crosses arranged in an N×N grid, detects their centres, and converts pixel coordinates to physical offsets for error analysis.

Method A1- Automatic (OpenCV)
1. Convert images to grayscale and apply a binary threshold
2. Use Canny edge detection to find the edges of crosses and Hough line transform to detect line segments along the edges
3. Apply a filter which selects lines found on the central cross of each image
4. Find the average of each vertical and horizontal line using a robust mean and find the centre of the cross using the intersection.
5. Repeat for all images and then normalise by subtracting the origin from each point

Method A2- Manual (OpenCV)
1. Automatically zoom in on the central cross 
2. User manually clicks on the centre of the cross to provide pixel coordinates
3. Repeat for all images and normalise by the origin

Method B- Convert from pixels to mm
1. Uses a precalculated value to convert between the pixels of the photo taken by the camera to the physical distance in mm
2. Divides all pixel coordinates by this constant value

Method C- Correct camera rotation
1. Uses least-squares fitting to fit lines that go through every cross in the same column and row as the origin point.
2. These rows should have negligible deviation from the axis, and any deviation is assumed to be due to a rotation because of camera misalignment.
3. By calculating the gradient of each line, we can find the angle to rotate all the coordinates by to remove the camera rotation.
4. We can use a matrix to adjust the positions for camera rotation because we do not want to account for this when adjusting the galvo input.
(This method requires the input tensor to have been converted to mm already)


Part 2- Calibration
Method 1- Polynomial fitting
1. Input the offsets of the actual measured points from the commanded positions
2. Build polynomial features ( degree=3: [1, x, y, x², xy, y², x³, x²y, xy², y³] ) using least-squares fitting to find x and y parameters.


Method 2 -Modifying Inputs
1. Requires x and y parameters to have been found.
2. Takes a txt file containing commanded coordinates and applies the polynomial mapping to correct the galvo inputs and saves the modified file

