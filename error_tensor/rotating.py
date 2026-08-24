import numpy as np

def offsets_to_positions(offset_array, grid_size, spacing):

    rows = grid_size
    columns = grid_size

    if columns % 2 == 1:  # Odd number of columns
        x = (np.arange(columns) - (columns - 1) / 2) * spacing
    else:  # Even number of columns
        x = (np.arange(columns) - columns / 2 + 0.5) * spacing
        
    # Calculate y coordinates centered at 0
    # Reverse the order so that row 0 has positive y (top)
    if rows % 2 == 1:  # Odd number of rows
        y = -(np.arange(rows) - (rows - 1) / 2) * spacing
    else:  # Even number of rows
        y = -(np.arange(rows) - rows / 2 + 0.5) * spacing
        
    # Create meshgrid
    xx, yy = np.meshgrid(x, y, indexing='xy')
        
    # Stack to create (2, rows, columns) tensor
    grid = np.stack([xx, yy], axis=0)

    positions = grid + offset_array

    return positions, grid

def angle_to_level(npy_array, origin_position):
    """
    Find rotation angle to level the cross pattern using least squares.
    
    Args:
        npy_filepath: Path to .npy file with shape (2, N, N) containing x,y coords
        origin_position: NxN position of origin point in the grid
    
    Returns:
        rotation_angle: Angle in degrees to rotate the entire grid
    """

    xaxis_data = npy_array[:,origin_position, :]
    yaxis_data = npy_array[:,:, origin_position] 


    # Fit horizontal line (x-axis data)
    # We want to fit: x = m * col + b (should be close to horizontal)
    # But we're checking the y-values along the horizontal line
    # y = m * index + b, ideally m = 0

    coeffs_horizontal = np.polyfit(xaxis_data[1], xaxis_data[0], 1)
    m_h = coeffs_horizontal[0]  # slope of horizontal line


    # Fit vertical line (y-axis data)
    # x = m * index + b, ideally m = 0

    coeffs_vertical = np.polyfit(yaxis_data[1], yaxis_data[0], 1)
    m_v = coeffs_vertical[0]  # slope of vertical line

    # Calculate rotation angle
    # The angle of the horizontal line from horizontal
    angle_h = np.arctan(m_h) - np.pi/2 
    print(angle_h)
    # The angle of the vertical line from vertical
    angle_v = np.arctan(m_v)
    print(angle_v)

    # Average the two angles (they should be similar)
    rotation_angle = (angle_h + angle_v) / 2
    print(f"rotation angle = {rotation_angle}")
    return rotation_angle

def rotate_data(data, angle):
    """
    Rotate all coordinates by angle (radians).
    
    Args:
        data: (2, N, N) array where data[0] = x coords, data[1] = y coords
        angle: Rotation angle in radians
    
    Returns:
        rotated_data: (2, N, N) array with rotated coordinates
    """
    x = data[0]
    y = data[1]
    
    # Rotation matrix
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    
    # Apply rotation: x' = x*cos - y*sin, y' = x*sin + y*cos
    x_rot = x * cos_theta - y * sin_theta
    y_rot = x * sin_theta + y * cos_theta
    
    rotated_data = np.array([x_rot, y_rot])
    
    return rotated_data