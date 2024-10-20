import os
import random
import shutil
from collections import defaultdict
from tqdm import tqdm  # tqdm 라이브러리 임포트

def sampling_FromTrainDataSet__():
    image_dir = r'/home/started/ultralytics/datasets/train/images'  # 훈련 이미지 경로
    label_dir = r'/home/started/ultralytics/datasets/train/labels'  # 훈련 라벨 경로
    output_image_dir = r'/home/started/ultralytics/datasets/val/images'  # 샘플링된 검증 이미지 저장 경로
    output_label_dir = r'/home/started/ultralytics/datasets/val/labels'  # 샘플링된 검증 라벨 저장 경로
    #코랩 디렉터리

    #image_dir = '/content/TLD_2024/train/images'
    #label_dir = '/content/TLD_2024/train/labels'
    #output_image_dir = '/content/TLD_2024/val/images'  # 샘플링된 검증 이미지를 저장할 경로
    #output_label_dir = '/content/TLD_2024/val/labels'  # 샘플링된 검증 라벨을 저장할 경로

    if os.path.exists(output_image_dir):
        shutil.rmtree(output_image_dir)  # 존재하면 디렉터리 삭제
    os.makedirs(output_image_dir, exist_ok=True)  # 새로 생성

    if os.path.exists(output_label_dir):
        shutil.rmtree(output_label_dir)  # 존재하면 디렉터리 삭제
    os.makedirs(output_label_dir, exist_ok=True)  # 새로 생성

    # 클래스별 이미지 저장할 딕셔너리
    class_to_images = defaultdict(list)

    # 이미지와 라벨 파일을 매칭해서 클래스별로 분류
    for img_file in os.listdir(image_dir):
        if img_file.endswith('.jpg'):  # 이미지 파일만 처리
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt'))

            # 라벨 파일 읽기
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        class_id = int(line.split()[0])  # 클래스 ID 추출
                        class_to_images[class_id].append((img_path, label_path))

    # 목표 샘플 수 #######################
    target_sample_count = 7000  # 검증 데이터셋에 대해 샘플링할 이미지 수
    total_classes = len(class_to_images)

    # 각 클래스에서 필요한 샘플 수 계산
    samples_per_class = max(1, target_sample_count // total_classes)

    sampled_images = []
    samples_per_class_count = defaultdict(int)  # 각 클래스에서 샘플링된 수 저장

    # 클래스별 이미지 수 출력
    total_images_per_class = {class_id: len(img_label_list) for class_id, img_label_list in class_to_images.items()}

    # 클래스별 샘플링 및 프로그레스 바
    for class_id, img_label_list in tqdm(class_to_images.items(), desc="Sampling classes", total=len(class_to_images)):
        # 클래스에서 샘플링할 이미지 수 계산
        num_samples = min(len(img_label_list), samples_per_class)

        sampled_list = random.sample(img_label_list, num_samples)
        sampled_images += sampled_list
        samples_per_class_count[class_id] += num_samples  # 샘플링 수 기록

    # 샘플링된 이미지를 새 폴더에 복사 및 프로그레스 바
    for i, (img_path, label_path) in tqdm(enumerate(sampled_images), desc="Copying images", total=len(sampled_images)):
        img_output_path = os.path.join(output_image_dir, f"{i}_{os.path.basename(img_path)}")
        label_output_path = os.path.join(output_label_dir, f"{i}_{os.path.basename(label_path)}")

        shutil.copy(img_path, img_output_path)  # 이미지 파일 복사
        shutil.copy(label_path, label_output_path)  # 라벨 파일 복사

    # 전체 샘플링 수 출력
    print(f'Total sampled images and labels copied: {len(sampled_images)} out of {target_sample_count}.')

    # 각 클래스에서 총 이미지 수와 샘플링된 수 출력
    for class_id in total_images_per_class.keys():
        total_images = total_images_per_class[class_id]
        samples = samples_per_class_count[class_id]
        print(f'Class {class_id}: Total Images = {total_images}, Samples = {samples}')
sampling_FromTrainDataSet__()


