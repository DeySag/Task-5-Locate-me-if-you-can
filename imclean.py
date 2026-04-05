import cv2
import numpy as np

# 1. Load the image
map_img = cv2.imread('/home/map.png') # Update with your exact filename
hsv_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2HSV)

# 2. Broaden the HSV net
# Lowering Saturation (S) and Value (V) catches the washed-out and darker edge pixels
lower_green = np.array([35, 20, 20]) 
upper_green = np.array([85, 255, 255])

# Create the initial mask
green_mask = cv2.inRange(hsv_img, lower_green, upper_green)

# 3. The Magic Step: Dilate the mask
# This creates a 3x3 grid to expand our mask by 1 pixel in every direction
kernel = np.ones((3,3), np.uint8)
dilated_mask = cv2.dilate(green_mask, kernel, iterations=1)

# 4. Replace the aggressively masked pixels with white
map_img[dilated_mask > 0] = (255, 255, 255)

# Save the final pristine map
cv2.imwrite('clean_map.png', map_img)
print("Aggressive cleanup complete! Check clean_map.png")