# 클래스 별로 이미지 분류
# 클래스 별 이미지 수량 출력
from started.augmentation import classificationImage as cf

cf.start()

# 해당 클래스의 이미지 증강 3번3
from started.augmentation.clsimageaug import class13aug as c13
for i in range(0,3):
    c13.start()
