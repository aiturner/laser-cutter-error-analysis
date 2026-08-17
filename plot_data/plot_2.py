import numpy as np
import matplotlib.pyplot as plt

def datapoints_from_txt(file):
    """
    Extracts first, second and third columns from txt file.
    If first column is zero, remove that row.
    After this, remove the first column entirely.
    """
    x_points = []
    y_points = []
    
    with open(file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 3:
                    col1 = float(parts[0])
                    col2 = float(parts[1])
                    col3 = float(parts[2])
                    
                    # Only keep rows where first column is NOT zero
                    if abs(col1) > 1e-6:
                        x_points.append(col2)
                        y_points.append(col3)
    
    datapoints = np.column_stack([x_points, y_points])
    return datapoints

def plot_ori_and_mod_same_axis(file, modified_file, step=5):
    """
    Extracts data from both file and modified_file using datapoints_from_txt,
    then plots both on the same axis.
    
    Parameters:
    file : str - path to original file
    modified_file : str - path to modified file
    step : int - plot every nth point (default: 5)
    """
    # Extract data from both files
    original_points = datapoints_from_txt(file)
    modified_points = datapoints_from_txt(modified_file)
    
    # Subsample every 'step' point
    original_subsampled = original_points[::step]
    modified_subsampled = modified_points[::step]
    
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Plot both on same axis
    plt.scatter(original_subsampled[:, 0], original_subsampled[:, 1], 
                s=1, alpha=0.5, color='blue', label='Original')
    plt.scatter(modified_subsampled[:, 0], modified_subsampled[:, 1], 
                s=1, alpha=0.5, color='red', label='Modified')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Original vs Modified Points (every {step}th point shown)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    txt1 = input("Input relative filepath of first txt (I_O_data/GalvoCrossGrid_9.txt) :")
    txt2 = input("Input relative filepath of second txt (I_O_data/GalvoCrossGrid_9_modified.txt) :")
    plot_ori_and_mod_same_axis(txt1, 
                               txt2, step=10)