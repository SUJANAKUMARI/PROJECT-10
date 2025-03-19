import cv2
import os

# Define source and destination folders
source_folder = r"C:\Users\Sujana\Desktop\FINAL PROJECT10\val original image"
destination_folder = r"C:\Users\Sujana\Desktop\FINAL PROJECT10\data\images\val"

# Ensure destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Track if images were found
image_found = False

for filename in os.listdir(source_folder):
    file_path = os.path.join(source_folder, filename)
    
    if os.path.isfile(file_path):
        image = cv2.imread(file_path)
        
        if image is None:
            print(f"⚠️ Skipping {filename} (Not an image or cannot be read)")
            continue
        
        image_found = True
        resized_image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        save_path = os.path.join(destination_folder, filename)
        cv2.imwrite(save_path, resized_image)
        print(f"✅ Resized and saved: {filename}")

if not image_found:
    print("❌ No images were found in the source folder. Check the folder path!")

print("🎯 Image resizing process completed.")