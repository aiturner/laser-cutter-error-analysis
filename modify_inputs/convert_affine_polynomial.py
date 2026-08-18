import numpy as np 
import pandas as pd

def calibrate_galvo_polynomial(commanded, offset_csv_path, degree=2):
    """
    Calibrate galvo system using polynomial fitting to capture non-affine distortion.
    
    The calibration finds a polynomial that maps commanded coordinates to correction offsets:
        [ex, ey] ≈ P(gx, gy)
    
    Parameters
    ----------
    commanded : (N, 2) array
        Commanded positions [gx, gy]
    offsets : (N, 2) array
        Measured offsets [ex, ey]
    degree : int
        Polynomial degree (1=affine, 2=quadratic, 3=cubic, etc.)
    
    Returns
    -------
    params_x : array
        Polynomial coefficients for x-offset
    params_y : array
        Polynomial coefficients for y-offset
    residuals : (N, 2) array
        Offsets - fitted_offsets
    stats : dict
        Statistics about the fit
    """
    df = pd.read_csv(offset_csv_path)
    offsets = df[['x', 'y']].values
    offsets = np.asarray(offsets, dtype=float)
    #offsets = offsets * 1/398.64
    #commanded = commanded.transpose(1, 2, 0).reshape(-1, 2)
    #offsets = offsets.reshape(-1, 2)

    # Convert to numpy arrays
    commanded = np.asarray(commanded, dtype=float)
    #offsets = np.asarray(offsets, dtype=float)
    #print(f"COMMANDED{commanded}")
    #print(f"OFFSETS = {offsets}")
    # Input validation
    if commanded.ndim != 2 or commanded.shape[1] != 2:
        raise ValueError("commanded must have shape (N, 2)")
    if offsets.shape != commanded.shape:
        raise ValueError("offsets must have the same shape as commanded")
    if commanded.shape[0] < 3:
        raise ValueError("Need at least 3 points for fit")
    
    N = commanded.shape[0]
    gx = - commanded[:, 0]
    gy = commanded[:, 1]
    
    # Build polynomial features
    # degree=1: [1, x, y]
    # degree=2: [1, x, y, x², xy, y²]
    # degree=3: [1, x, y, x², xy, y², x³, x²y, xy², y³]
    features = []
    for d in range(degree + 1):
        for i in range(d + 1):
            j = d - i
            features.append((gx ** i) * (gy ** j))
    
    M = np.column_stack(features)
    
    # Fit x-offset: ex = P(gx, gy)
    params_x, residuals_x, rank_x, s_x = np.linalg.lstsq(M, offsets[:, 0], rcond=None)
    
    # Fit y-offset: ey = P(gx, gy)
    params_y, residuals_y, rank_y, s_y = np.linalg.lstsq(M, offsets[:, 1], rcond=None)
    
    # Calculate fitted offsets and residuals
    fitted_offsets = np.column_stack([
        M @ params_x,
        M @ params_y
    ])
    residuals = offsets - fitted_offsets
    
    stats = {
        'rms_error_x': np.sqrt(np.mean(residuals[:, 0]**2)),
        'rms_error_y': np.sqrt(np.mean(residuals[:, 1]**2)),
        'rms_error_total': np.sqrt(np.mean(residuals**2)),
        'max_error_x': np.max(np.abs(residuals[:, 0])),
        'max_error_y': np.max(np.abs(residuals[:, 1])),
        'num_points': N,
        'degree': degree,
        'num_params': len(params_x),
    }
    
    return params_x, params_y, residuals, stats


def convert_positions_polynomial(commanded, params_x, params_y, degree=2):
    """
    Convert commanded positions to corrected positions using polynomial fit.
    
    Parameters
    ----------
    commanded : (N, 2) array
        Commanded positions [gx, gy]
    params_x : array
        Polynomial coefficients for x-offset
    params_y : array
        Polynomial coefficients for y-offset
    degree : int
        Polynomial degree used in calibration
    
    Returns
    -------
    corrected : (N, 2) array
        Corrected positions
    """
    commanded = np.asarray(commanded, dtype=float)
    gx = commanded[:, 0]
    gy = commanded[:, 1]
    
    # Build polynomial features
    features = []
    for d in range(degree + 1):
        for i in range(d + 1):
            j = d - i
            features.append((gx ** i) * (gy ** j))
    
    M = np.column_stack(features)
    #print(M)
    # Calculate corrections
    corrections = np.column_stack([
        M @ params_x,
        M @ params_y
    ])
    
    #print(f"CORRECTIONS{corrections}")
    # Apply corrections
    corrected = commanded + corrections
    
    #print(corrected)

    return corrected

def extract_from_txt(file):
    xpositions = []
    ypositions = []
    
    with open(file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 3:
                    xpositions.append(float(parts[1]))
                    ypositions.append(float(parts[2]))
    
    # Return as (N, 2) - each row is a point [x, y]
    return np.column_stack([xpositions, ypositions])

def np_to_csv(numpy_array_path, csv_output_path='output.csv'):
    tensor = np.load(numpy_array_path)
    #print(tensor)

    reshaped = tensor.reshape(2, -1)
    df = pd.DataFrame(reshaped.T, columns=['x', 'y'])
    df.to_csv(csv_output_path, index=False)
    return csv_output_path

def create_commanded_grid(rows, columns, spacing):
    """
    Creates a 3D tensor holding x and y coordinates of a rows x columns size grid spaced by spacing.
    The grid is centered at (0,0) with negative and positive coordinates.
    Y increases upward (positive above origin, negative below).
    
    Parameters
    ----------
    rows : int
        Number of rows in the grid (y direction)
    columns : int
        Number of columns in the grid (x direction)
    spacing : float
        Spacing between grid points in both x and y directions
    
    Returns
    -------
    grid : (rows, columns, 2) array
        3D tensor where grid[i, j] = [x, y] coordinate
        x increases with columns (j) to the right
        y increases with rows (i) going upward (positive y at top)
        The grid is centered at (0,0)
    """
    
    # Calculate x coordinates centered at 0
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
    
    #print(grid)
    
    return grid

def replace_txt(file, new_points):
    """
    Replace coordinates in a text file with new points.
    
    Parameters:
    file : str
        Input file path
    new_points : (N, 2) array
        Each row is [x, y] coordinate
    """
    with open(file, 'r') as f:
        lines = f.readlines()
    
    # Create output filename
    output_file = file.replace('.txt', '_modified.txt')
    
    with open(output_file, 'w') as f:
        for i, line in enumerate(lines):
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 3 and i < new_points.shape[0]:  # Check rows, not columns
                    # Replace 2nd and 3rd columns (index 1 and 2)
                    parts[1] = f"{new_points[i, 0]:.4f}"  # new x
                    parts[2] = f"{new_points[i, 1]:.4f}"  # new y
                    f.write(' '.join(parts) + '\n')
                else:
                    # Keep original line if it doesn't have enough columns
                    f.write(line)
            else:
                f.write(line)
    
    print(f"Modified file saved as: {output_file}")

def rotate_npy_to_level(input_path):
    """
    Rotate points so the leftmost and rightmost points have the same y coordinate.
    Saves as {input_path}_rotated.npy
    """
    # Load the tensor
    tensor = np.load(input_path)
    
    # Convert to (N, 2) points
    if tensor.ndim == 3 and tensor.shape[0] == 2:
        points = tensor.transpose(1, 2, 0).reshape(-1, 2)
    else:
        points = tensor.reshape(-1, 2)
    
    # Find leftmost and rightmost points
    min_idx = np.argmin(points[:, 0])
    max_idx = np.argmax(points[:, 0])
    p1 = points[min_idx]
    p2 = points[max_idx]
    
    # Calculate rotation angle
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = -np.arctan2(dy, dx)
    
    # Rotate
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = points @ rot_matrix.T
    
    # Keep centroid in same place
    centroid = np.mean(points, axis=0)
    new_centroid = np.mean(rotated, axis=0)
    rotated = rotated + (centroid - new_centroid)
    
    # Save
    output_path = input_path.replace('.npy', '_rotated.npy')
    np.save(output_path, rotated)
    print(f"Rotated by {np.degrees(angle):.4f} degrees, saved to {output_path}")
    
    return output_path


def angle_to_level(npy_array, origin_position):
    """
    Find rotation angle to level the cross pattern using least squares.
    
    Args:
        npy_filepath: Path to .npy file with shape (2, N, N) containing x,y coords
        origin_position: NxN position of origin point in the grid
    
    Returns:
        rotation_angle: Angle in degrees to rotate the entire grid
    """
    
    # Load data
    
    # Get the row and column at the origin
    xaxis_data = npy_array[:,origin_position, :]
    yaxis_data = npy_array[:,:, origin_position] 
    #print(xaxis_data)

    #print("\n")
    #print(yaxis_data)

    # ---- Fit horizontal line (x-axis data) ----
    # We want to fit: x = m * col + b (should be close to horizontal)
    # But we're checking the y-values along the horizontal line
    # They should be constant (y = origin_y) if level
    
    # Fit line to yaxis_data vs indices (should be horizontal)
    # y = m * index + b, ideally m = 0
    coeffs_horizontal = np.polyfit(xaxis_data[1], xaxis_data[0], 1)
    m_h = coeffs_horizontal[0]  # slope of horizontal line
    #print(f"m_h= {m_h}")


    # ---- Fit vertical line (y-axis data) ----
    # Fit line to xaxis_data vs indices (should be vertical)
    # x = m * index + b, ideally m = 0
    coeffs_vertical = np.polyfit(yaxis_data[1], yaxis_data[0], 1)
    m_v = coeffs_vertical[0]  # slope of vertical line
    #print(f"m_v= {m_v}")
    # ---- Calculate rotation angle ----
    # The angle of the horizontal line from horizontal
    angle_h = - (np.pi / 2) - np.arctan(m_h)
    #print(f"angle_h= {angle_h}")

    # The angle of the vertical line from vertical
    # For vertical line, slope m means it's rotated by arctan(m) from vertical
    angle_v = np.arctan(m_v)
    #print(f"angle_v (radians)= {angle_v}")

    # Average the two angles (they should be similar)
    # Note: sign might need adjustment depending on convention
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

if __name__ == "__main__":
    #rotated_path = rotate_npy_to_level('output_tensor.npy')
    csv_path = np_to_csv('output_tensor.npy', 'output_tensor.csv')
    commanded = create_commanded_grid(9, 9, 5)
    print(F"COMMANDED{commanded}")
    commanded = commanded.transpose(1, 2, 0).reshape(-1, 2)

    # TO DO rotate raw data to correct for camera rotation before calibration
    # Use least squares method to fit lines to x and y axis and use this to find rotation.
    
    xpara, ypara, residuals, stats = calibrate_galvo_polynomial(commanded, "output_tensor.csv" , degree=3)

    print(f"XPARA= {xpara}")
    print(f"YPARA= {ypara}")

    points = extract_from_txt("modify_inputs/GalvoCrossGrid_9.txt")
    new_points = convert_positions_polynomial(points, xpara, ypara, degree=3)
    replace_txt("modify_inputs/GalvoCrossGrid_9.txt", new_points)