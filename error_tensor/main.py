import numpy as np
import pandas as pd
import re
import os

from center_coords_linefitting import find_cross_center
from tensor_from_directory import add_positions

def pixels_to_mm(tensor, k = 398.642):
    """
    Convert pixel offsets to physical offsets.

    """
    
    return tensor * 1/k 

def parse_image_name(image_name):
    """
    Inputs:
    Image name of form 'n (m).bmp'

    Outputs:
    Row = n
    Column = m

    """
    name = image_name.split('.')[0]
    name = name.strip()

    pattern = r'^(\d+)\s*\(\s*(\d+)\s*\)$'
    match = re.search(pattern, name)
    
    if not match:
        raise ValueError(f"Image name '{image_name}' does not match pattern 'n (m)'")
    
    row = int(match.group(1))
    column = int(match.group(2))
    
    return row, column

def build_tensor():
    """
    Interactive function to build a calibration tensor.
    Asks user for grid size, origin position, and allows adding points or saving.
    
    Returns:
        numpy array: Calibration tensor of shape (2, rows, cols)
    """

    print("\n STEP 1: Define Grid Size")
    while True:
        try:
            rows = int(input("  Number of rows (N): "))
            cols = int(input("  Number of columns (M): "))
            if rows > 0 and cols > 0:
                break
            print("Grid size must be positive integers. Try again.")
        except ValueError:
            print("Please enter valid integers.")

    # Initalise a 3 dimentional tensor with size that user inputs
    tensor = np.full((2, rows, cols), -1, dtype=np.float32)

    print("\n STEP 2: Set Origin Position")
    # Sets the origin position, this position will have zero offset due to lense distortion 
    # This point can be used to give a relative zero error position for the other points

    while True:
        try:
            origin_row = int(input(f"  Origin row (1-{rows}): "))
            origin_col = int(input(f"  Origin col (1-{cols}): "))
            if 1 <= origin_row <= rows and 1 <= origin_col <= cols:
                break
            print(f"Row must be 1-{rows}, Col must be 1-{cols}. Try again.")
        except ValueError:
            print(" Please enter valid integers.")

    print(f"Origin set at position ({origin_row}, {origin_col}) with coordinates (0, 0)")

    print("STEP 3: Add Points or Save Tensor")

    while True:
        print("\nOptions:")
        print("1  Add a offsets from images")
        print("2  Save tensor and exit")
        print("3  Show current tensor")
        print("4  Convert from pix to mm")
        print("5  Exit without saving")


        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            # Give the user an option of offset detection method.

            print("Method 1 : Automatic Detection, \nMethod 2: Manual Detection")
            choice2 = input("\nSelect option (1-2):")

            if choice2 == '1':
                directory_path = input("  Enter directory path of all images with names n(m): ").strip()

                # For each image in the directory it sets the pixel position of the center and saves it to the tensor
                for image_file in os.listdir(directory_path):
                    row, col = parse_image_name(image_file)
                    image_path = os.path.join(directory_path, image_file)
                    print(row)
                    print(col)
                    center = find_cross_center(image_path, debug = False)
                    if center:
                        x, y = center
                        tensor[0, row-1, col-1] = x
                        tensor[1, row-1, col-1] = y


                np.save("absolute_pos_pix1.npy" ,tensor)

                reshaped = tensor.reshape(2, -1).T
                df = pd.DataFrame(reshaped, columns=['x-opencv', 'y-opencv'])
                df.to_csv('absolute_pos_pix1.csv', index=False)

                # Corrects all positions by subtracting the pixel position of the origin point center
                tensor[0,:,:] = tensor[0,:,:] - tensor[0,origin_row-1,origin_col-1]
                tensor[1,:,:] = tensor[1,:,:] - tensor[1,origin_row-1,origin_col-1]
                
            elif choice2 == '2':
                image_directory = input("Enter directory path of all images with names n(m):")
                origin_point = f"{origin_row} ({origin_col})"
                tensor = add_positions(tensor, image_directory , origin_point)

            else:
                print("Invalid Option")

        elif choice == '2':
            # Saves the tensor
            np.save("offset_data.npy", tensor)
            print("Tensor saved to offset_data.npy")
            return tensor

        elif choice == '3':

            print("X coordinates:\n")
            print(tensor[0, :, :])
            print("\nY coordinates:\n")
            print(tensor[1, :, :])

        elif choice == '4':
            # Converts the tensor from pixel position to millimeters using a precalculated constant
            tensor = pixels_to_mm(tensor)

        elif choice == '5':
            return tensor
        
        else:
            print("Invalid Option")
            

if __name__ == "__main__":
    tensor = build_tensor()
    # outputs a 2,N,N Numpy array containing data about the offsets of each point, 
    # conventions of 
    # x is positive to the right
    # y is positive down