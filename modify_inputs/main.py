from convert_affine_polynomial import calibrate_galvo_polynomial , convert_positions_polynomial , extract_from_txt , np_to_csv , create_commanded_grid , replace_txt

if __name__ == "__main__":
    npy_filename = input("Filename of the npy array containing offset data (offset_data.npy) : ")
    csv_path = np_to_csv(npy_filename , 'offset_data.csv')
    grid_size = int(input("NxN Gridsize: "))
    grid_spacing = int(input("Spacing between two crosses on grid in mm: "))

    commanded = create_commanded_grid(grid_size, grid_size, grid_spacing)
    commanded_reshape = commanded.transpose(1, 2, 0).reshape(-1, 2)

    # TO DO rotate raw data to correct for camera rotation before calibration
    # Use least squares method to fit lines to x and y axis and use this to find rotation.
    
    xpara, ypara, residuals, stats = calibrate_galvo_polynomial(commanded_reshape, "offset_data.csv" , degree=3)

    print(f"XPARA= {xpara}")
    print(f"YPARA= {ypara}")
    points_relfilepath = input("txt Relative filepath containing points to be modified (I_O_data/GalvoCrossGrid_9.txt) : ")
    points = extract_from_txt(points_relfilepath)
    new_points = convert_positions_polynomial(points, xpara, ypara, degree=3)
    replace_txt(points_relfilepath, new_points)