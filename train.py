import os
import torch
from ultralytics import YOLO
import gc
from utils import setPath as p
os.environ["NCCL_P2P_DISABLE"] = "1"

gc.collect()
torch.cuda.empty_cache()

model = YOLO(p.get_p("trainModel/tld_detection/weights/best.pt"))
model.train(cfg=p.get_p("commoms/cfg.yaml"))
