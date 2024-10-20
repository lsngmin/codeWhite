from ultralytics import YOLO
import os
from utils import setPath as p

os.environ["NCCL_P2P_DISABLE"] = "1"

model = YOLO(p.get_p("trainModel/tld_detection/weights/best.pt"))
data = p.get_p()+ "/commons/tld_2024.yaml"

metrics = model.val(
    data=data,
    batch=22,
    imgsz=1280,
    device="0,1",
    workers=48,
    verbose=True)
# metrics.box.map  # map50-95
# metrics.box.map50  # map50
