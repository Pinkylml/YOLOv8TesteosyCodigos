from Segmentacion import Segmentacion
from deteccionCrestas import DeteccionCrestas
import cv2
import os
#current_dir = os.getcwd()
#print(current_dir)
path_img='/media/jeffersonc/DiscoRespaldo/YOLOv8/images/1695339635519segmented3.png'
imagen=cv2.imread(path_img)
segmentacion1=Segmentacion(imagen)
imagen_segmentada,isAFInger=segmentacion1.recortarDedoLocalizado()
if bool(isAFInger):
    deteccion1=DeteccionCrestas(imagen_segmentada)
    flag,imagen_filtrada=deteccion1.detectar()
    if bool(flag):
        print("Hay Crestas")
    else:
        print("No Hay crestas")
else:
    print("No hay dedos")

