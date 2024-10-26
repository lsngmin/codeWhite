import class13aug as c
# 증강을 수행할 폴더 경로
if_ = [
    "/home/codeWhite/ultralytics/datasets/train_class/class_13/images",
]
lf_ =   [ 
	"/home/codeWhite/ultralytics/datasets/train_class/class_13/labels",
]


# 이미지와 라벨 증강 수행
for j, k in zip(if_, lf_) :
    c.start(j, k)
    print("succ!!")
