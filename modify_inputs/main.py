from convert_affine_polynomial import calibrate_galvo_polynomial , convert_positions_polynomial , angle_to_level, rotate_data, extract_from_txt , np_to_csv , create_commanded_grid , replace_txt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    npy_filepath = input("Filename of the npy array containing offset data (offset_data.npy) : ")
    if npy_filepath.endswith('.csv'):
        offset_data = pd.read_csv(npy_filepath).values
        M = int(np.sqrt(offset_data.shape[0]))
        offset_data = offset_data.T.reshape(2, M, M)
    elif npy_filepath.endswith('.npy'):
        offset_data = np.load(npy_filepath)
    else:
        raise ValueError("File must be .csv or .npy")

    grid_size = int(input("NxN Gridsize: "))
    grid_spacing = int(input("Spacing between two crosses on grid in mm: "))
    origin_position = int(input("NxN Origin Position: "))

    commanded = create_commanded_grid(grid_size, grid_size, grid_spacing)
    commanded_reshape = commanded.transpose(1, 2, 0).reshape(-1, 2)

    positions = offset_data + commanded

    angle = - angle_to_level(positions, origin_position)
    positions_data_rotated = rotate_data(positions, angle)
    offset_data_rotated = positions_data_rotated - commanded
    np.save("offset_data_rotated.npy", offset_data_rotated)

    print(offset_data)
    print("\n")
    print(offset_data_rotated)

    csv_path = np_to_csv("offset_data_rotated.npy" , 'offset_data.csv')
    xpara, ypara, residuals, stats = calibrate_galvo_polynomial(commanded_reshape, "offset_data.csv" , degree=3)

    print(f"XPARA= {xpara}")
    print(f"YPARA= {ypara}")
    points_relfilepath = input("txt Relative filepath containing points to be modified (I_O_data/GalvoCrossGrid_9.txt) : ")
    points = extract_from_txt(points_relfilepath)
    new_points = convert_positions_polynomial(points, xpara, ypara, degree=3)
    replace_txt(points_relfilepath, new_points)
    grid_size = int(input("NxN Gridsize: "))
    grid_spacing = int(input("Spacing between two crosses on grid in mm: "))
    origin_position = int(input("NxN Origin Position: "))

    
    commanded = create_commanded_grid(grid_size, grid_size, grid_spacing)
    commanded_reshape = commanded.transpose(1, 2, 0).reshape(-1, 2)

    # Create the grid we measured by combining offsets and commanded positons
    offset_data = np.load(npy_filepath)
    positions = offset_data + commanded

    # Rotates the positions so that the x and y axis are horisontal and vetical
    # This corrects for any error due to the rotation in the camera which we do not want to account for
    angle = - angle_to_level(positions, origin_position)
    positions_data_rotated = rotate_data(positions, angle)
    offset_data_rotated = positions_data_rotated - commanded
    np.save("offset_data_rotated.npy", offset_data_rotated)

    # Converts to csv
    csv_path = np_to_csv("offset_data_rotated.npy" , 'offset_data.csv')

    # Fits xpara and ypara required to map the commanded coordinates to the actual postions
    # This is approximately the mapping caused by the systematic errors such as the lense
    # degree=3: [1, x, y, x², xy, y², x³, x²y, xy², y³]
    xpara, ypara, residuals, stats = calibrate_galvo_polynomial(commanded_reshape, "offset_data.csv" , degree=3)
    
    print("[1, x, y, x², xy, y², x³, x²y, xy², y³]")
    print(f"XPARA= \n {xpara}")

    print(f"YPARA= \n {ypara}")

    # Used to change the galvo coordinates
    points_relfilepath = input("txt Relative filepath containing points to be modified (I_O_data/GalvoCrossGrid_9.txt) : ")
    points = extract_from_txt(points_relfilepath)
    new_points = convert_positions_polynomial(points, xpara, ypara, degree=3)
    replace_txt(points_relfilepath, new_points)