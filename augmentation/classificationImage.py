import os
import shutil
from ..utils import setPath
def start(p, j) :
    # 이미지 및 라벨 경로 설정
    image_dir = p
    label_dir = j

    output_dir = p.get_p("datasets") + "train_class"
    os.makedirs(output_dir, exist_ok=True)

    # 이미지 및 라벨 파일 분류 및 클래스별 저장
    for img_file in os.listdir(image_dir):
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
            # 라벨 파일에서 클래스 정보 추출
            label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
            src_label_path = os.path.join(label_dir, label_file)

            if os.path.exists(src_label_path):
                with open(src_label_path, 'r') as f:
                    lines = f.readlines()
                    class_ids = set([line.split()[0] for line in lines])  # 라벨 파일의 모든 줄에서 클래스 ID 추출

                    for class_id in class_ids:
                        class_name = f'class_{class_id}'

                        # 클래스별 폴더 설정
                        class_image_output_dir = os.path.join(output_dir, class_name, 'images')
                        class_label_output_dir = os.path.join(output_dir, class_name, 'labels')

                        # 클래스별 폴더가 없으면 생성
                        os.makedirs(class_image_output_dir, exist_ok=True)
                        os.makedirs(class_label_output_dir, exist_ok=True)

                        # 이미지를 클래스별 폴더로 복사 (동일 이미지가 여러 클래스에 포함될 수 있음)
                        src_image_path = os.path.join(image_dir, img_file)
                        dst_image_path = os.path.join(class_image_output_dir, img_file)
                        if not os.path.exists(dst_image_path):
                            shutil.copy(src_image_path, dst_image_path)

                        # 라벨 파일도 동일한 클래스 폴더로 복사
                        dst_label_path = os.path.join(class_label_output_dir, label_file)
                        if not os.path.exists(dst_label_path):
                            shutil.copy(src_label_path, dst_label_path)

    print('Images and labels have been classified and saved by class.')

    total_images = 0
    for class_folder in os.listdir(output_dir):
        class_image_dir = os.path.join(output_dir, class_folder, 'images')
        if os.path.isdir(class_image_dir):
            num_images = len([name for name in os.listdir(class_image_dir) if name.endswith('.jpg') or name.endswith('.png')])
            total_images += num_images
            print(f'Class {class_folder} contains {num_images} images.')

    print(f'Total number of images: {total_images}')
