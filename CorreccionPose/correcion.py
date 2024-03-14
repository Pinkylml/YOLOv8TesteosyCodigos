import cv2
import numpy as np



from ultralytics import YOLO
# Cargar la imagen del dedo (asegúrate de que ya está segmentado)
imagen_dedo = cv2.imread('/media/jeffersonc/DiscoRespaldo/YOLOv8/images/1695339635519.JPEG')
# Load a model
model = YOLO('/media/jeffersonc/DiscoRespaldo/YOLOv8/CorreccionPose/yolov8m-pose.pt')  # load an official model


# Predict with the model
results = model(imagen_dedo,show=True,save=True)  # predict on an image
