import os
import torch
from ultralytics import YOLO
import cv2
import warnings

warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")
os.environ["NCCL_P2P_DISABLE"] = "1"

# YOLO 모델 로드
model = YOLO('/Users/smin/Desktop/ultralytics/tests/runs/detect/best.pt')

# 테스트할 이미지가 있는 디렉토리 경로
image_dir = r"/Users/smin/Desktop/ultralytics/datasets/TLD_2024/test/images"

# 라벨 파일이 있는 경로 (예: 라벨 파일이 다른 디렉토리에 있을 경우)
label_dir = r"/Users/smin/Desktop/ultralytics/datasets/TLD_2024/test/labels"


# 원래 라벨을 불러오는 함수 (YOLO 형식으로 저장되어 있다고 가정)
def load_ground_truth_labels(label_path):
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                labels.append((int(class_id), x_center, y_center, width, height))
    return labels


# 디렉토리 내 모든 이미지 파일 불러오기
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# 이미지들을 한 장씩 테스트
for img_file in image_files:
    # 이미지 경로 생성
    img_path = os.path.join(image_dir, img_file)

    # 라벨 파일 경로 생성 (라벨 파일이 이미지 파일과 동일한 이름이므로 확장자만 변경)
    label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt'))

    # 이미지 로드
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # 원래 라벨 불러오기 (라벨 파일이 있는 경우)
    ground_truth_labels = load_ground_truth_labels(label_path)

    # 이미지 사이즈 가져오기
    img_h, img_w = img.shape[:2]

    # 원래 라벨이 있는 경우 시각화
    if ground_truth_labels:
        for gt_class_id, x_center, y_center, width, height in ground_truth_labels:
            # YOLO 형식의 바운딩 박스를 픽셀 좌표로 변환
            x1 = int((x_center - width / 2) * img_w)
            y1 = int((y_center - height / 2) * img_h)
            x2 = int((x_center + width / 2) * img_w)
            y2 = int((y_center + height / 2) * img_h)

            # 원래 라벨의 바운딩 박스 그리기 (색상: 파란색, 두께: 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 파란색
            label = f"GT Class {gt_class_id}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    else:
        print(f"No ground truth labels found for {img_file}")

    # 예측 수행
    results = model(img)

    # 바운딩 박스 결과 추출
    boxes = results[0].boxes

    # 테스트 라벨(모델 예측) 그리기
    for box in boxes:
        # 박스의 클래스 ID와 확률값 가져오기
        class_id = int(box.cls.item())  # 클래스 ID
        conf = box.conf.item()  # 확률

        # 박스 좌표 가져오기
        xyxy = box.xyxy[0].cpu().numpy()  # 좌표

        # 좌표값을 정수로 변환
        x1, y1, x2, y2 = map(int, xyxy)

        # 테스트 라벨의 바운딩 박스 그리기 (색상: 초록색, 두께: 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 초록색
        label = f"Pred Class {class_id} Conf: {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 이미지 보기
    cv2.imshow("Predictions", img)

    # 사용자가 키를 누르면 다음 이미지로 넘어감
    key = cv2.waitKey(0)

    # 'q' 키를 누르면 중단
    if key == ord('q'):
        break

# 모든 창 닫기
cv2.destroyAllWindows()
