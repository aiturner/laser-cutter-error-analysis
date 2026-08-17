import cv2
import numpy as np
import re
import os

def add_positions(tensor, directory_path , origin_point):
    """ 
    Takes a directory of images each with name n_m where this is the row column of each point

    Finds pixelated position of each cross and subtracts he pixelated position of the origin point.

    Adds the points to the tensor.
    """
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    for image in files:
        row , column = parse_image_name(image)

        image_path = os.path.join(directory_path, image)
        xposition, yposition = manually_select_center(image_path, f"Click center of {image}")

        tensor[0, row-1, column-1] = xposition
        tensor[1, row-1, column-1] = yposition

    origin_image = os.path.join(directory_path, origin_point)
    origin_image = origin_image + ".bmp"

    xorigin, yorigin = manually_select_center(origin_image, "Click the ORIGIN cross center")
    
    row, column = parse_image_name(origin_point)
    tensor[0, row-1, column-1] = xorigin
    tensor[1, row-1, column-1] = yorigin

    tensor[0, :, :] -= xorigin
    tensor[1, :, :] -= yorigin

    return tensor


def parse_image_name(image_name):
    """
    Inputs:
    Image name of form 'n (m).bmp'

    Outputs:
    Row = n
    Column = m

    """
    name = image_name.split('.')[0]
    name = name.strip()

    pattern = r'^(\d+)\s*\(\s*(\d+)\s*\)$'
    match = re.search(pattern, name)
    
    if not match:
        raise ValueError(f"Image name '{image_name}' does not match pattern 'n (m)'")
    
    row = int(match.group(1))
    column = int(match.group(2))
    
    return row, column

def find_center_cross(image_path, min_area=100):
    """
    Find the center of the cross closest to image center.
    
    Returns:
        (x, y): Coordinates of the closest-to-center cross
    """
    # Read and process
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(cv2.GaussianBlur(gray, (5,5), 0), 100, 255, cv2.THRESH_BINARY)

    binary_filled = binary.copy()
    h, w = binary.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(binary_filled, mask, (0, 0), 255)
    holes = cv2.bitwise_not(binary_filled)
    binary_final = cv2.bitwise_or(binary, holes)

    contours, _ = cv2.findContours(binary_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get image center
    h, w = img.shape[:2]
    img_center = (w/2, h/2)
    
    # Find centers of all contours and pick closest to image center
    closest = None
    min_dist = float('inf')
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            M = cv2.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                dist = np.hypot(cx - img_center[0], cy - img_center[1])
                if dist < min_dist:
                    min_dist = dist
                    closest = (cx, cy)
    return closest

def manually_select_center(image_path, window_name="Click the cross center"):
    """
    Display image and let user click the center of the cross.
    
    Returns:
        (x, y): Coordinates of the clicked point
        None: If user presses 'q' or 'ESC' to cancel
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    approx_x, approx_y = find_center_cross(image_path)
    
    h, w = img.shape[:2]
    print(f"Original image dimensions: width={w}, height={h}")
    print(f"Approximate center: ({approx_x}, {approx_y})")
    
    zoom_factor = 2
    crop_size = 1000
    half_crop = crop_size // 2
    
    x_start = max(0, approx_x - half_crop)
    x_end = min(w, approx_x + half_crop)
    y_start = max(0, approx_y - half_crop)
    y_end = min(h, approx_y + half_crop)
    
    print(f"Crop window: x_start={x_start}, x_end={x_end}, y_start={y_start}, y_end={y_end}")
    print(f"Crop window size: width={x_end - x_start}, height={y_end - y_start}")
    print(f"Cross position relative to crop: x={approx_x - x_start}, y={approx_y - y_start}")
    
    cropped = img[y_start:y_end, x_start:x_end]
    
    zoomed = cv2.resize(cropped, (crop_size * zoom_factor, crop_size * zoom_factor), 
                       interpolation=cv2.INTER_CUBIC)
    
    selected_point = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_point
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x = x_start + (x / zoom_factor)
            orig_y = y_start + (y / zoom_factor)
            selected_point = (orig_x, orig_y)
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        display = zoomed.copy()
        
        if selected_point is not None:
            cx, cy = selected_point
            zoom_x = int((cx - x_start) * zoom_factor)
            zoom_y = int((cy - y_start) * zoom_factor)
            cv2.drawMarker(display, (zoom_x, zoom_y), (0, 255, 0), 
                          cv2.MARKER_CROSS, 20, 2)
        
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:
            cv2.destroyWindow(window_name)
            return None
        
        if selected_point is not None:
            cv2.waitKey(500)
            break
    
    cv2.destroyWindow(window_name)
    return selected_point