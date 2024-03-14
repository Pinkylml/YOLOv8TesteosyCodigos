from ultralytics import YOLO
import os
import cv2
import errno
# Load a model
#model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('/media/jeffersonc/DiscoRespaldo/YOLOv8/DeteccionCrestasYOLOv8_30epoch.pt')  # load a custom model
path='/media/jeffersonc/DiscoRespaldo/YOLOv8/images/PruebaNovalida.png'
image=cv2.imread(path)
results = model(path, conf=0.3) 
for r in results:
    boxes=r.boxes
    cls=boxes.cls
    valor=cls[0].item()
    if valor==1.0:
        print(valor)
