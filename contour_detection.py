import cv2

def find_circle_center(image_path):
    """
    Find the center of a hollow circle by detecting the largest contour
    and computing its centroid.
    
    Args:
        image_path: Path to the image file
    
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
    # You might need to adjust the threshold value (127) based on your image
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in the image")
    
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate centroid using image moments
    moments = cv2.moments(largest_contour)
    if moments['m00'] == 0:
        raise ValueError("Contour area is zero")
    
    center_x = int(moments['m10'] / moments['m00'])
    center_y = int(moments['m01'] / moments['m00'])
    
    # Print results
    print(f"Center coordinates: ({center_x}, {center_y})")
    #print(f"Contour area: {cv2.contourArea(largest_contour):.2f} pixels")
    #print(f"Number of contour points: {len(largest_contour)}")
    return center_x , center_y

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "/Users/arthurturner/Documents/Projects/lz/Test1x1.jpg"  # Change this to your image file
    
    try:
        # Method 1: Contour-based (works for any hollow shape)
        center_x, center_y = find_circle_center(image_path)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if the image path is correct")
        print("2. Adjust the threshold value (127) in the code")
        print("3. Ensure the circle is clearly visible against the background")
        print("4. Try using the Hough method for perfect circles")