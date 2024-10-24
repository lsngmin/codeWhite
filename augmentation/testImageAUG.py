import os
import cv2
import numpy as np
import albumentations as A
import shutil

import os
import cv2
import numpy as np


def adjust_color(image, hue_shift=11, saturation_increase_for_black=10, brightness_increase_for_black=20, min_brightness=20, brightness_increase_for_red=0):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split the channels (Hue, Saturation, Value)
    h, s, v = cv2.split(hsv)

    # Ensure all channels have the same data type (e.g., uint8)
    h = h.astype(np.uint8)
    s = s.astype(np.uint8)
    v = v.astype(np.uint8)

    # 1. 빨간색 계열의 색조(Hue)를 조정
    # Create two masks for red hue ranges: 0~10 and 170~180 (for red)
    mask_red1 = cv2.inRange(h, 0, 10)  # 첫 번째 빨간색 범위
    mask_red2 = cv2.inRange(h, 170, 180)  # 두 번째 빨간색 범위

    # Combine the two red masks
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Adjust hue only for red colors
    h = np.where(mask_red > 0, (h + hue_shift) % 180, h)

    # 2. 검정색 계열의 채도(Saturation)와 명도(Value)를 조정
    # 검정색 범위를 확장 (Saturation: 0~100, Value: 0~100)
    mask_black = cv2.inRange(s, 0, 100) & cv2.inRange(v, 0, 100)

    # Very dark areas (v < 50), increase brightness first before saturation
    very_dark_mask = v < min_brightness

    # Adjust brightness for very dark pixels
    v = np.where(very_dark_mask, np.clip(v + brightness_increase_for_black, 0, 255), v)

    # Adjust saturation and brightness for black pixels
    s = np.where(mask_black > 0, np.clip(s + saturation_increase_for_black, 0, 255), s)

    v = np.where(mask_red > 0, np.clip(v + brightness_increase_for_red, 0, 255), v)

    # Merge the adjusted channels back and convert to BGR
    hsv_adjusted = cv2.merge([h, s, v])

    # Convert back to BGR color space
    adjusted_image = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

    return adjusted_image
def process_all_images(input_dir, output_dir):
    """
    Apply the image adjustments to all images in a directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_exts = [".jpg", ".png", ".bmp"]

    for img_file in os.listdir(input_dir):
        if any(img_file.lower().endswith(ext) for ext in img_exts):
            img_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, img_file)

            # Load the image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Unable to read {img_path}")
                continue

            # Adjust the image
            adjusted_image = adjust_color(image)

            # Save the adjusted image
            cv2.imwrite(output_path, adjusted_image)
            print(f"Adjusted image saved to {output_path}")
