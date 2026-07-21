import numpy as np
#import cv2
#import os
import re
from pathlib import Path

from contour_detection import find_circle_center

def parse_image_name(filename):
    """
    Parse image filename to extract grid position.
    Expected format: circle_nxm.png (e.g., circle_3x4.png)
    
    Args:
        filename: Image filename
    
    Returns:
        tuple: (row, col) or None if pattern doesn't match
    """
    # Match pattern like circle_3x4.png, circle_1x2.png, etc.
    pattern = r'circle_(\d+)x(\d+)\.(png)$'
    match = re.search(pattern, filename, re.IGNORECASE)
    
    if match:
        row = int(match.group(1))
        col = int(match.group(2))
        return row, col
    return None

def process_grid_images(directory_path, verbose=True):
    """
    Process all circle_nxm.png images in a directory and create a tensor.
    """
    # Find all circle_nxm.png files
    image_files = list(Path(directory_path).glob('circle_*.png'))
    
    if not image_files:
        print(f"No circle_nxm.png files found in {directory_path}")
        return None
    
    # Parse filenames to get grid positions
    positions = {}
    for img_file in image_files:
        pos = parse_image_name(img_file.name)
        if pos:
            positions[pos] = str(img_file)
        elif verbose:
            print(f"Skipping {img_file.name} - doesn't match pattern circle_nxm.png")

    # Determine grid size
    rows = max([p[0] for p in positions.keys()])
    cols = max([p[1] for p in positions.keys()])
    
    if verbose:
        print(f"Grid size detected: {rows}×{cols}")
        print(f"Found {len(positions)} images out of {rows*cols} possible")
    
    # Initialize tensor with -1 for missing positions
    tensor = np.full((2, rows, cols), -1, dtype=np.float32)
    processed = {}
    missing = []

    # Process each image
    for (row, col), img_path in positions.items():
        if verbose:
            print(f"\nProcessing circle_{row}x{col}.png...")
        
        try:
            center_x, center_y = find_circle_center(img_path)
            
            # Store coordinates (adjust for 0-indexing)
            tensor[0, row-1, col-1] = center_x
            tensor[1, row-1, col-1] = center_y
            processed[(row, col)] = (center_x, center_y)
            
            if verbose:
                print(f"({row},{col}) -> ({center_x}, {center_y})")
                
        except Exception as e:
            if verbose:
                print(f"Failed: {str(e)}")
            missing.append((row, col))

    # Check for missing positions
    for r in range(1, rows+1):
        for c in range(1, cols+1):
            if (r, c) not in positions:
                missing.append((r, c))
    
    if missing and verbose:
        print(f"\n Missing images at positions: {missing}")

            # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY:")
    print(f"  Grid size: {rows}×{cols}")
    print(f"  Total positions: {rows*cols}")
    print(f"  Successfully processed: {len(processed)}")
    print(f"  Missing/failed: {len(missing)}")
    print(f"  Tensor shape: {tensor.shape}")
    
    return {
        'tensor': tensor,
        'grid_size': (rows, cols),
        'processed_positions': processed,
        'missing_positions': missing
    }


if __name__ == "__main__":
    # Set your directory path
    directory = "/Users/arthurturner/Documents/Projects/laser_cutter_accuracy/images_circle_nxm"
    #output_dir = "./output/"
    
    # Create output directory
    #os.makedirs(output_dir, exist_ok=True)
    
    # ===== METHOD 1: Auto-detect grid size from filenames =====
    #print("\n" + "="*50)
    #print("METHOD 1: Auto-detect grid size")
    #print("="*50)
    
    result = process_grid_images(directory, verbose=True)
    
    if result:
        tensor = result['tensor']
    
        # Print the tensor to the terminal
        print("\nTensor shape:", tensor.shape)
        print("X coordinates (layer 0):\n", tensor[0, :, :])
        print("Y coordinates (layer 1):\n", tensor[1, :, :])
    
        # Save as .npy file (for later use)
        np.save('output_tensor.npy', tensor)
        print("\nTensor saved to output_tensor.npy")
    else:
        print("No results to display")