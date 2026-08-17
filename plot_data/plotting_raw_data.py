import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_grid(grid_size=9, spacing=5):
    """
    Create a chessboard grid of points.
    
    Parameters:
    -----------
    grid_size : int
        Number of points per axis (e.g., 9 = 9x9 grid)
    spacing : float
        Spacing between grid points
    
    Returns:
    --------
    grid_points : (N, 2) array
        Array of grid coordinates
    """
    half = (grid_size - 1) / 2
    x = np.arange(-half, half + 1) * spacing
    y = np.arange(-half, half + 1) * spacing
    xx, yy = np.meshgrid(x, y)
    
    # Reshape to (N, 2)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    return grid_points


def load_csv_as_array(csv_file):
    """
    Load x and y columns from CSV and return as (N, 2) array.
    """
    df = pd.read_csv(csv_file)
    points = df[['x', 'y']].values
    return points


def plot_data_with_grid(csv_file1, csv_file2, theta=0.00000000001, grid_size=9, spacing=5, 
                        colors=None, labels=None ):
    """
    Plot original grid and two modified datasets on the same axis.
    """
    
    # Default colors and labels
    if colors is None:
        colors = ['black', 'blue', 'red']
    
    if labels is None:
        labels = ['Original Grid', 'Original_measuremts', 'Modified measurements']
    
    # Step 1: Create the grid
    grid_points = create_grid(grid_size, spacing)
    
    # Step 2: Load CSV data
    data1 = load_csv_as_array(csv_file1)
    data2 = load_csv_as_array(csv_file2)
    data1 += grid_points
    data2 += grid_points
    data2 = rotate(data2, theta)

    # Step 3: Combine grid with data (if needed, or keep separate for plotting)
    # For plotting, we'll plot them separately with different colors
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot original grid (black dots)
    ax.scatter(grid_points[:, 0], grid_points[:, 1], 
              color=colors[0], 
              label=labels[0],
              s=30,
              alpha=0.5,
              marker='s')
    
    # Plot first dataset
    ax.scatter(data1[:, 0], data1[:, 1], 
              color=colors[1], 
              label=labels[1],
              alpha=0.7,
              s=50)
    
    # Plot second dataset
    ax.scatter(data2[:, 0], data2[:, 1], 
              color=colors[2], 
              label=labels[2],
              alpha=0.7,
              s=50)
    
    # Formatting
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Grid and Modified Data Comparison')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.show()

def rotate(points, theta):
    """
    Rotate points by theta radians about the origin.
    
    Parameters:
    -----------
    points : (N, 2) array
        Array of points to rotate
    theta : float
        Rotation angle in radians (positive = counterclockwise)
    
    Returns:
    --------
    rotated : (N, 2) array
        Rotated points
    """
    # Create rotation matrix
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
    
    # Apply rotation to all points
    rotated = points @ rot_matrix.T
    
    return rotated

def find_theta_csv(input_file):
    """
    Load CSV with x,y columns, rotate all points about origin, save to new CSV.
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    output_file : str
        Path to output CSV file
    angle_degrees : float
        Rotation angle in degrees (positive = counterclockwise)
    """
    # Load
    df = pd.read_csv(input_file)
    points = df[['x', 'y']].values
    
    dy = points[36,1] - points[44,1]
    print(dy)
    print(points[36,1])
    print(points[44,1])
    theta = np.arctan(dy / 40)
    return theta

if __name__ == "__main__":
    #theta = find_theta_csv("output_tensor_Test5.csv")
    plot_data_with_grid("offset_data.csv","offset_data")