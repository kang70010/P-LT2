from ultralytics import YOLO
import numpy as np

yolo = YOLO("yolov8-plt.yaml")

yolo.train(
    data='BHBDATAset.yaml',
    epochs=300,
    batch=60,
    workers=10,
    lr0=0.001,
    optimizer='AdamW',
    seed=0,
)
