import cv2
import numpy as np
import math

map_img = cv2.imread('clean_map.png')
if map_img is None:
    print("Error: Could not find clean_map.png. Check the filename!")
    exit()

height, width = map_img.shape[:2]

# --- VERIFIED TRANSFORMATION PARAMETERS ---
ROTATION_DEG = 60.0   
SCALE = 12.5          
OFFSET_X = 560
OFFSET_Y = 625
# ------------------------------------------

path_coords = []
print("Parsing FLASER data from aces.clf...")

# Extract Coordinates from FLASER lines
with open('/home/aces.clf', 'r') as file:
    for line in file:
        tokens = line.strip().split()
        if not tokens:
            continue
            
        if tokens[0] == 'FLASER':
            # In a FLASER line, the number of laser readings is the second token
            num_readings = int(tokens[1])
            
            # The X and Y coordinates are stored immediately AFTER the laser array
            # Format: FLASER [num] [val1...valN] [X] [Y] [Theta] ...
            x = float(tokens[2 + num_readings])
            y = float(tokens[3 + num_readings])
            path_coords.append((x, y))

print(f"Extracted {len(path_coords)} trajectory points.")

# Apply Global Transformation
theta = math.radians(ROTATION_DEG)
cos_t = math.cos(theta)
sin_t = math.sin(theta)

pixel_coords = []
for x, y in path_coords:
    # Rotate the raw meter coordinates around the robot's start origin
    rot_x = (x * cos_t) - (y * sin_t)
    rot_y = (x * sin_t) + (y * cos_t)
    
    # Convert to pixels and apply the manual offsets
    px = int((rot_x * SCALE) + OFFSET_X)
    
    # Flip the Y-axis for image coordinates (Top-Left Origin)
    py = int(height - ((rot_y * SCALE) + OFFSET_Y))
    
    pixel_coords.append([px, py])

# Render the Result
pts = np.array(pixel_coords, np.int32).reshape((-1, 1, 2))

# Draw the trajectory in bright red
cv2.polylines(map_img, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

# Save the output
output_path = '/root/final_localized_map_flaser.png'
cv2.imwrite(output_path, map_img)
print(f"Success! Map saved to {output_path}")