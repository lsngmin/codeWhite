import albumentations as A
import cv2
import os
from glob import glob
import numpy as np
import uuid  # UUID 모듈 추가
import random

# 증강 파이프라인 설정
def get_augmentation_pipeline():
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0), contrast_limit=(-0.2, 0.2), p=0.5),
        A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=(-50, 0), p=0.5),
        A.RandomGamma(gamma_limit=(60, 100), p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.MotionBlur(p=0.3),
        A.GridDistortion(p=0.05),
        A.HorizontalFlip(p=0.19),  # 좌우 반전
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=10, p=0.5)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


# 증강을 적용하는 함수
def start(input_image_folder, input_label_folder):
    # 증강 파이프라인 생성
    random.seed()
    augmentations = get_augmentation_pipeline()

    # 이미지 경로와 라벨 경로 가져오기
    image_paths = glob(os.path.join(input_image_folder, "*.jpg"))
    original_image_count = len(image_paths)
    augmented_image_count = 0

    for img_path in image_paths:
        # 이미지 로드
        image = cv2.imread(img_path)
        if image is None:
            continue

        # 라벨 파일 경로 생성 (이미지 파일과 동일한 이름을 가진 .txt 파일)
        base_name = os.path.basename(img_path).replace(".jpg", ".txt")
        label_path = os.path.join(input_label_folder, base_name)

        # 바운딩 박스 및 클래스 라벨 가져오기
        bboxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(int(class_id))

        # 이미지 및 바운딩 박스 증강 적용
        augmented = augmentations(image=image, bboxes=bboxes, class_labels=class_labels)

        # 고유한 파일명 만들기 (중복 방지, UUID 사용)
        unique_filename = f"aug_{uuid.uuid4().hex}.jpg"  # UUID를 파일명으로 사용
        input_img_path = os.path.join(input_image_folder, unique_filename)

        # 증강된 이미지 저장
        cv2.imwrite(input_img_path, augmented['image'])

        # 증강된 라벨 저장
        input_label_path = os.path.join(input_label_folder, unique_filename.replace(".jpg", ".txt"))
        with open(input_label_path, 'w') as output_label_file:
            for bbox, class_id in zip(augmented['bboxes'], class_labels):
                x_center, y_center, width, height = bbox
                output_label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        augmented_image_count += 1

    # 최종 출력
    print(f"총 {original_image_count}개의 원본 이미지가 있고, {augmented_image_count}개의 증강된 이미지가 생성되었습니다.")
    print(f"전체 파일 수: {original_image_count + augmented_image_count}")



