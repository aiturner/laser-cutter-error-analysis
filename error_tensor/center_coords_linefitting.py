import cv2
import numpy as np
def get_center_line(line_group):
    """
    Find center line by:
    1. Split lines into upper/lower groups based on initial y estimate
    2. Split each group into left/right based on initial x estimate
    3. Average all 4 groups to find center point
    """
    def robust_mean(values):
        """Remove outliers using IQR method (most common statistical approach)"""
        if len(values) == 0:
            return 0
    
        sorted_vals = np.sort(values)
        q1 = np.percentile(sorted_vals, 25)
        q3 = np.percentile(sorted_vals, 75)
        iqr = q3 - q1
    
        # Define outlier bounds (1.5 * IQR is standard)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
    
        # Filter out outliers
        filtered = [v for v in values if lower_bound <= v <= upper_bound]
    
        return np.mean(filtered) if filtered else np.mean(values)
    
    if not line_group:
        return None
    
    # Check if lines are more horizontal or vertical
    sample_line = line_group[0]
    dx = sample_line[2] - sample_line[0]
    dy = sample_line[3] - sample_line[1]
    initial_y = int(robust_mean([line[1] for line in line_group]))
    initial_x = int(robust_mean([line[0] for line in line_group]))

    if abs(dx) >= abs(dy):
        # More horizontal - use y-values for vertical split, x-values for horizontal split
        upper_group = [line for line in line_group if line[1] < initial_y]
        lower_group = [line for line in line_group if line[1] > initial_y]
        
        # Edge case: if all lines are on one side
        if not upper_group or not lower_group:
            # Fall back to simple average
            avg_x = int(robust_mean([line[0] for line in line_group]))
            avg_y = int(robust_mean([line[1] for line in line_group]))
            return (avg_x, avg_y, avg_x, avg_y)
        
        # Step 3: Split upper and lower groups into left/right based on x
        upper_left = [line for line in upper_group if line[0] < initial_x]
        upper_right = [line for line in upper_group if line[0] > initial_x]
        lower_left = [line for line in lower_group if line[0] < initial_x]
        lower_right = [line for line in lower_group if line[0] > initial_x]

        upper_left_avg_y = int(robust_mean([line[1] for line in upper_left])) 
        upper_right_avg_y = int(robust_mean([line[1] for line in upper_right])) 
        lower_left_avg_y = int(robust_mean([line[1] for line in lower_left]))
        lower_right_avg_y = int(robust_mean([line[1] for line in lower_right]))
                
        # Average the two group averages
        center_y = int((upper_left_avg_y + upper_right_avg_y +lower_left_avg_y +lower_right_avg_y ) / 4)
                
        # Average x-coordinates from all lines
        avg_x1 = int(robust_mean([line[0] for line in line_group]))
        avg_x2 = int(robust_mean([line[2] for line in line_group]))
        return (avg_x1, center_y, avg_x2, center_y)
    else:
        # More vertical - use x-values for horizontal split, y-values for vertical split
        left_group = [line for line in line_group if line[0] < initial_x]
        right_group = [line for line in line_group if line[0] > initial_x]
        
        # Edge case: if all lines are on one side
        if not left_group or not right_group:
            # Fall back to simple average
            avg_x = int(robust_mean([line[0] for line in line_group]))
            avg_y = int(robust_mean([line[1] for line in line_group]))
            return (avg_x, avg_y, avg_x, avg_y)
        
        # Split left and right groups into upper/lower based on y
        left_upper = [line for line in left_group if line[1] < initial_y]
        left_lower = [line for line in left_group if line[1] > initial_y]
        right_upper = [line for line in right_group if line[1] < initial_y]
        right_lower = [line for line in right_group if line[1] > initial_y]
        
        # Average x-values for each group
        left_upper_avg_x = int(robust_mean([line[0] for line in left_upper])) 
        left_lower_avg_x = int(robust_mean([line[0] for line in left_lower])) 
        right_upper_avg_x = int(robust_mean([line[0] for line in right_upper]))
        right_lower_avg_x = int(robust_mean([line[0] for line in right_lower]))
        
        # Average the four group averages
        center_x = int((left_upper_avg_x + left_lower_avg_x + right_upper_avg_x + right_lower_avg_x) / 4)
        
        # Average y-coordinates from all lines
        avg_y1 = int(robust_mean([line[1] for line in line_group]))
        avg_y2 = int(robust_mean([line[3] for line in line_group]))
        return (center_x, avg_y1, center_x, avg_y2)

def find_cross_center(image_path, debug = False):

    """
    Find the center of the central cross in an image using Hough Line Transform.
    """

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image")
        return None
    
    height, width = img.shape[:2]
    
    # Preprocessing with binary threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply binary threshold to make cross solid white on black background
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # Find edges on the binary image 
    edges = cv2.Canny(binary, 50, 150)
    
    # Detect lines using Hough Transform 
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=30,     
        minLineLength=20,  
        maxLineGap=15      
    )
    
    if lines is None:
        print("No lines detected")
        return None
    
    # Filter lines to keep only those in the center box (only detects the central cross)
    center_x, center_y = width // 2, height // 2
    search_radius_x = width // 6
    search_radius_y = height // 6
    
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        if abs(mx - center_x) < search_radius_x and abs(my - center_y) < search_radius_y:
            filtered_lines.append((x1, y1, x2, y2))
    
    if len(filtered_lines) < 2:
        print("Not enough lines found on central cross")
        return None
    
    # Separate lines into horizontal and vertical groups
    horizontal = []
    vertical = []
    
    for x1, y1, x2, y2 in filtered_lines:
        dx = x2 - x1
        dy = y2 - y1
        angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
        
        # Set tolerance for vertical and horisontal lines.
        if angle < 5 or angle > 175:  # Was 20/160
            horizontal.append((x1, y1, x2, y2))
        elif 85 < angle < 95:  # Was 70-110
            vertical.append((x1, y1, x2, y2))
    
    if not horizontal or not vertical:
        print("Need both horizontal and vertical lines")
        return None
    
    # 5. Get center lines using the iterative method
    best_h = get_center_line(horizontal)
    best_v = get_center_line(vertical)
    
    if best_h is None or best_v is None:
        return None
    
    # Compute intersection point of the horisontal and vertical arms of the cross
    def line_intersection(line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        
        return int(px), int(py)
    
    center = line_intersection(best_h, best_v)
    
    if center is None:
        print(f"Lines are parallel, no intersection{image_path}")
        return None
    
    # Draw detection results
    if debug:
        debug_img = img.copy()
    
    # Draw all filtered lines (green)
        for x1, y1, x2, y2 in filtered_lines:
            cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw best horizontal (blue) and vertical (red)
        if best_h:
            cv2.line(debug_img, (best_h[0], best_h[1]), (best_h[2], best_h[3]), (255, 0, 0), 3)
            cv2.line(debug_img, (best_v[0], best_v[1]), (best_v[2], best_v[3]), (0, 0, 255), 3)
        if best_v:
    
    # Draw center point (magenta)
            cv2.circle(debug_img, center, 8, (255, 0, 255), -1)
            cv2.circle(debug_img, center, 15, (255, 0, 255), 2)
    
    # Draw the center box region
            cv2.rectangle(debug_img, 
                     (center_x - search_radius_x, center_y - search_radius_y),
                     (center_x + search_radius_x, center_y + search_radius_y),
                     (255, 255, 0), 2)
    
            cv2.imwrite('detection_result.jpg', debug_img)
            print("Detection result saved to 'detection_result.jpg'")
    
    return center


# Example usage
if __name__ == "__main__":

    result = find_cross_center("/Users/arthurturner/Documents/Projects/laser_cutter_accuracy/Test 6/1 (1).bmp", debug=True)
    
    if result:
        print(f"Cross center found at: ({result[0]}, {result[1]})")
    else:
        print("Could not find cross center")