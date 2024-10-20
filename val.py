from ultralytics import YOLO
import os

os.environ["NCCL_P2P_DISABLE"] = "1" 

path_lc = "/Users/smin/Desktop/ultralytics/codeWhite/"
path_sv =  "/home/codeWhite/ultralytics/started/"
path_g = path_lc if str(os.getcwd())[1] == 'U' else path_sv
model_filename = path_g + "trainModel/tld_detection/weights/best.pt"
model = YOLO(model_filename)

data = path_g + "tld_2024.yaml"

metrics = model.val(
    data=data,
    batch=22,
    imgsz=1280,
    device="0,1",
    workers="48",
    verbose=True)
# metrics.box.map  # map50-95
# metrics.box.map50  # map50
