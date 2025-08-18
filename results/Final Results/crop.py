import cv2
import sys

# Load the image
image_path = "resunet/prediction_result_4.png"  # Use relative or absolute path
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

if image is None:
    sys.exit(f"❌ Failed to load image at {image_path}. Please check the path or file format.")

# Get image dimensions
height, width = image.shape[:2]

# Define size of crop box (690x690)
box_size = 340

# Compute starting x-coordinate for the rightmost third
third_width = width // 3
right_third_start = 2 * third_width
center_x = right_third_start + (third_width // 2)

# Calculate crop coordinates centered in the rightmost third
left_start = max(0, center_x - (box_size // 2)) - 71
right_end = min(width, left_start + box_size) + 0

# Centered vertically
top_start = max(0, (height // 2) - (box_size // 2)) +1
bottom_end = min(height, top_start + box_size) + 3

# Crop the image
cropped_image = image[top_start:bottom_end, left_start:right_end]

# Ensure cropped image is not empty
if cropped_image.size == 0:
    sys.exit("❌ Cropped image is empty. Check coordinates.")

# Save the cropped image (lossless PNG)
output_path = "sample5_resunet.png"
success = cv2.imwrite(output_path, cropped_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

if success:
    print(f"✅ Cropped image saved as '{output_path}'")
else:
    sys.exit("❌ Failed to save cropped image.")
