from ultralytics import YOLO
import torch
import cv2
#from PIL import Image
# Load a model
#model = YOLO('yolov8n-seg.pt')  
model = YOLO('/media/jeffersonc/DiscoRespaldo/YOLOv8/YOLOv8Segmentation50epoch.pt')  # load a custom model
path_img='/media/jeffersonc/DiscoRespaldo/YOLOv8/images/cedula2.png'
img=cv2.imread(path_img)
# Predict with the model
results = model(img,imgsz=640,show=True,save=True,save_crop=True,conf=0.2) 


# Show the results
"""for r in results:
    #print(r.boxes)
    numby=r.boxes
    """
""""print(numby)
    numpy2=numby.xyxy
    print(numpy2)
    coordenadas=numpy2.numpy()
    print(coordenadas)"""
"""confidenses=numby.conf
    maxarg=torch.argmax(confidenses)
    print(maxarg)   
    box=r.boxes
    box=box[maxarg]
    box=box.xyxy
    coord=box.numpy()
    print(coord)"""

