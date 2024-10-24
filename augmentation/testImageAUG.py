import os
import cv2
import numpy as np
import albumentations as A
import shutil

import os
import cv2
import numpy as np


def adjust_color(image, hue_shift=11, saturation_increase_for_black=10, brightness_increase_for_black=20, min_brightness=20, brightness_increase_for_red=0):

    # 1. 밝기 조정: 현재 이미지의 밝기를 낮추고 명암비를 높임
    alpha = 1.2  # 명암비를 높임 (1.0 이상으로 설정)
    beta = -50   # 밝기를 낮춤 (값을 음수로 설정하여 밝기를 낮춤)
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 2. Grayscale 변환: 이미지를 흑백으로 변환
    grayscale_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)

    # 3. Gaussian Blurring: 노이즈 제거를 위해 가우시안 블러링 적용
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

    # 4. 이미지를 HSV 색 공간으로 변환
    # 원본 이미지를 HSV로 변환 (흑백 이미지 대신 원본 이미지 사용)
    hsv_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2HSV)

    # HSV 이미지의 각 채널(Hue, Saturation, Value)을 분리하여 필요 시 조작 가능
    h, s, v = cv2.split(hsv_image)

    # Ensure all channels have the same data type (e.g., uint8)
    h = h.astype(np.uint8)
    s = s.astype(np.uint8)
    v = v.astype(np.uint8)

    # 1. 빨간색 계열의 색조(Hue)를 조정
    # 빨간색 범위의 마스크 생성 (0~10 및 170~180의 범위)
    mask_red1 = cv2.inRange(h, 0, 10)
    mask_red2 = cv2.inRange(h, 170, 180)

    # 두 마스크를 결합하여 빨간색 검출
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # 빨간색 영역에서 색조를 조정
    h = np.where(mask_red > 0, (h + hue_shift) % 180, h)

    # 2. 검정색 계열의 채도(Saturation) 및 명도(Value) 조정
    mask_black = cv2.inRange(s, 0, 100) & cv2.inRange(v, 0, 100)
    v = np.where(mask_black > 0, np.clip(v + brightness_increase_for_black, 0, 255), v)
    s = np.where(mask_black > 0, np.clip(s + saturation_increase_for_black, 0, 255), s)

    # 수정된 HSV 채널을 다시 병합하고 BGR로 변환
    hsv_adjusted = cv2.merge([h, s, v])
    final_adjusted_image = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

    return final_adjusted_image


def process_all_images(input_dir, output_dir):
    """
    Apply the image adjustments to all images in a directory.
    모든 이미지를 전처리하여 저장하는 함수.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_exts = [".jpg", ".png", ".bmp"]

    for img_file in os.listdir(input_dir):
        if any(img_file.lower().endswith(ext) for ext in img_exts):
            img_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, img_file)

            # 이미지 로드
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Unable to read {img_path}")
                continue

            # 이미지 조정
            final_adjusted_image = adjust_color(image)

            # 조정된 이미지 저장
            cv2.imwrite(output_path, final_adjusted_image)
            print(f"Adjusted image saved to {output_path}")
