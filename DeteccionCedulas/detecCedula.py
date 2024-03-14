from ultralytics import YOLO
import os
import cv2
import errno
# Load a model
#model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('/media/jeffersonc/DiscoRespaldo/YOLOv8/DeteccionCedulas/Det_Cedulas_Anv_Rev.pt') 
def testImages(path_img):

    path=path_img
    image=cv2.imread(path)
    results = model(path, conf=0.5,save=True) 
    cv2.imshow("Imagen",image)
    for r in results:
        boxes=r.boxes
        if (boxes):
            print("Es cedula")
            cls=boxes.cls
            
            valor=cls[0].item()
            conf=boxes.conf
            print(conf)
            print("Clases",cls)
            if valor==0.0:
                print("Digital")
            elif valor==1.0:
                print("Extranjera")
            elif valor==2.0:
                print("Amarilla")
            else:
                print("No es cedula")
        else:
            print("No es cedula")

path_images_reales="/media/jeffersonc/DiscoRespaldo/YOLOv8/DeteccionCedulas/Images"
print("Test Imagenes Reales")
print("Imagenes Cedulas")
for file_image in os.listdir(path_images_reales):
    extension = os.path.splitext(file_image)[1].lower()
    if extension in (".jpg", ".jpeg", ".png", ".bmp"):
        path_imagen=os.path.join(path_images_reales,file_image)
        testImages(path_imagen)
path_images_reales="/media/jeffersonc/DiscoRespaldo/YOLOv8/DeteccionCedulas/Images/No cedulas"
print("Imagenes No Cedulas")
for file_image in os.listdir(path_images_reales):
    extension = os.path.splitext(file_image)[1].lower()
    if extension in (".jpg", ".jpeg", ".png", ".bmp"):
        path_imagen=os.path.join(path_images_reales,file_image)
        testImages(path_imagen)