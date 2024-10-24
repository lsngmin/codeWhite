import os
import torch
import gc
from ultralytics import YOLO
from torch.utils.data import DataLoader
from utils import setPath as p
import cv2
from testImageAUG import adjust_color

os.environ["NCCL_P2P_DISABLE"] = "1"

# 메모리 정리
gc.collect()
torch.cuda.empty_cache()

# Custom Dataset 클래스 정의
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_list = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        image = cv2.imread(img_path)

        # 이미지 전처리 함수 호출
        processed_image = adjust_color(image)  # testImageAUG.py의 adjust_color 함수

        if self.transform:
            processed_image = self.transform(processed_image)

        return processed_image

# 모델 로드
model = YOLO(p.get_p("yolov10x.pt"))

# 데이터셋 경로 설정

img_dir = p.get_p("/home/codeWhite/ultralytics/datasets/train/images")  # 실제 이미지 경로로 변경
train_dataset = CustomDataset(img_dir=img_dir)

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=2)

# 모델 학습
model.train(data=p.get_p("commons/tld_2024.yaml"), epochs=50, train_loader=train_loader)

