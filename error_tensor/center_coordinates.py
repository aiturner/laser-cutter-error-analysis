import cv2

def find_all_centers(image_path, min_area=10):
    """
    Find the centers of ALL contours in the image

    Args:
        image_path: Path to the image file
        min_area: Minimum contour area to condsider
    
    Returns:
        tuple: (center_x, center_y) coordinates of the circle center
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in the image")
    
    # Filter contours by area and get their centers
    contour_data = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])
                contour_data.append((area, center_x, center_y))
    
    if len(contour_data) < 2:
        raise ValueError(f"Need at least 2 contours. Found {len(contour_data)}")
    
    # Sort by area (largest first) and take top 2
    contour_data.sort(key=lambda x: x[0], reverse=True)
    top_two = contour_data[:2]
    
    centers = [(x, y) for _, x, y in top_two]
    
    # Print results
    print(f"Found {len(contour_data)} contours total")
    print("Top 2 largest contours:")
    for i, (area, x, y) in enumerate(top_two):
        print(f"  Contour {i+1}: ({x}, {y}) with area {area:.0f}")
    
    return centers
def get_origin_and_target(centers, origin_position='upper'):
    """
    Determine which center is the origin based on position.
    
    Args:
        centers: List of 2 (x, y) tuples
        origin_position: 'upper', 'lower', 'left', or 'right'
    
    Returns:
        tuple: (origin_center, target_center)
    """
    if len(centers) != 2:
        raise ValueError(f"Expected 2 centers, got {len(centers)}")
    
    (x1, y1), (x2, y2) = centers
    
    # Determine which is which based on position
    if origin_position == 'upper':
        origin = (x1, y1) if y1 < y2 else (x2, y2)
        target = (x2, y2) if y1 < y2 else (x1, y1)
    elif origin_position == 'lower':
        origin = (x1, y1) if y1 > y2 else (x2, y2)
        target = (x2, y2) if y1 > y2 else (x1, y1)
    elif origin_position == 'left':
        origin = (x1, y1) if x1 < x2 else (x2, y2)
        target = (x2, y2) if x1 < x2 else (x1, y1)
    elif origin_position == 'right':
        origin = (x1, y1) if x1 > x2 else (x2, y2)
        target = (x2, y2) if x1 > x2 else (x1, y1)
    else:
        raise ValueError(f"Invalid origin_position: {origin_position}. Use 'upper', 'lower', 'left', or 'right'")
    
    return origin, target


def find_relative_coordinates(image_path, origin_position, min_area=10):
    """
    Find the two largest contours and return relative coordinates.
    
    Args:
        image_path: Path to the image file
        origin_position: 'upper', 'lower', 'left', or 'right'
        min_area: Minimum contour area to consider
    
    Returns:
        dict: {
            'origin': (x, y),
            'target': (x, y),
            'relative': (rel_x, rel_y),
            'all_centers': [(x1,y1), (x2,y2)]
        }
    """
    # Find the two largest centers
    centers = find_all_centers(image_path, min_area)
    
    # Determine which is origin
    origin, target = get_origin_and_target(centers, origin_position)
    
    # Calculate relative coordinates
    origin_x, origin_y = origin
    target_x, target_y = target
    rel_x = target_x - origin_x
    rel_y = target_y - origin_y
    
    print(f"\nOrigin ({origin_position}): {origin}")
    print(f"Target: {target}")
    print(f"Relative coordinates: ({rel_x}, {rel_y})")
    
    return {
        'origin': origin,
        'target': target,
        'relative': (rel_x, rel_y),
        'all_centers': centers
    }

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "/Users/arthurturner/Documents/Projects/lz/Test1x1.jpg"
    
    # Choose which is origin: 'upper', 'lower', 'left', or 'right'
    origin_position = 'upper'  # Change this as needed
    
    try:
        result = find_relative_coordinates(image_path, origin_position=origin_position)
        print(f"\nOrigin: {result['origin']}")
        print(f"Target: {result['target']}")
        print(f"Relative: ({result['relative'][0]}, {result['relative'][1]})")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if the image path is correct")
        print("2. Adjust the threshold value (127) in the code")
        print("3. Adjust min_area if needed")
        print("4. Make sure there are at least 2 visible contours")



# if entire crosses cannot fit on a single image, fit lines to the two arms on the cross and find intersection