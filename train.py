from ultralytics import YOLO
import numpy as np


yolo.train(
    data='BHBDATAset.yaml',
    epochs=300,
    batch=60,
    workers=10,
    lr0=0.001,
    optimizer='AdamW',
    seed=0,
)
