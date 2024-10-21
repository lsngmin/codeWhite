#!/bin/bash

# 복사할 파일의 경로
SOURCE_FOLDER="/home/codeWhite/ultralytics/started/trainModels/tld_detections_1021/results.csv"
TARGET_FOLDER="/home/codeWhite/ultralytics/started/rst/results.csv"

git pull origin master
rm -rf /home/codeWhite/ultralytics/started/rst/results.csv
cp $SOURCE_FOLDER $TARGET_FOLDER
# Git 저장소로 이동
cd /home/codeWhite/ultralytics/started/

# Git 상태 확인
git status

# Git 추가 (추가 또는 수정된 파일만)
git add . 

# 커밋 메시지 자동 생성
git commit -m "Auto-commit file $(basename $SOURCE_FOLDER) at $(date)"

# 원격 저장소에 푸시
git push origin master  # main 브랜치로 푸시, 필요시 브랜치 이름 변경

# 완료 메시지 출력
echo "File $(basename $SOURCE_FOLDER) copied and pushed to repository at $(date)"
