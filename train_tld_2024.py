import os
import torch
from ultralytics import YOLO
from torch.optim import RAdam
import gc
import sampling as S


gc.collect()
torch.cuda.empty_cache()
os.environ["NCCL_P2P_DISABLE"] = "1"
# 모델 로드
model = YOLO(r'/home/codeWhite/ultralytics/started/best.pt')
model.train(cfg=r"/home/codeWhite/ultralytics/started/cfg.yaml")
gc.collect()
torch.cuda.empty_cache()

