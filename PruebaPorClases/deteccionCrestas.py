from ultralytics import YOLO
from PIL import Image
import os
import errno
import cv2
import numpy as np

class DeteccionCrestas:
    def loadModel(self):
        directorio_actual = os.path.dirname(__file__) 
        path_model = os.path.join(directorio_actual, 'DeteccionCrestasYOLOv8_30epoch.pt')
        model = YOLO(path_model)
        return model
    
    def __init__(self, imagen):
        self.imagen_dedo_segmentado = imagen
        self.imagen_filtrada=None
        self.estatus=None
        #self.model_segmetation = self.__class__.loadModel()  # Accede al método de clase
        self.model_segmetation = self.loadModel()

    def segmentarDedo(self):
        results = self.model_segmetation(self.imagen_filtrada,imgsz=640) 
        return results
    
    def verificarCresta(self):
        flag=None
        crestas=False
        resultados=self.segmentarDedo()
        for r in resultados:
            boxes=r.boxes
            cls=boxes.cls
            valor=cls[0].item()
            if valor==0.0:
                print('se detectaron crestas')
                crestas=True
            else:
                print('No se detectaron crestas')
                crestas=False
        return crestas
        
    @staticmethod
    def binarization(imagen):
        image = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        # Binarización Adaptativa: Aplicar umbralización adaptativa
        imagen_binaria_adaptativa = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 2)

        # Extracción de Características: En este ejemplo, simplemente usaremos la imagen binaria adaptativa como característica
        caracteristicas = imagen_binaria_adaptativa

        # Plantilla de Huella Dactilar: La característica extraída puede ser la propia imagen binaria adaptativa
        plantilla = caracteristicas
        

        return plantilla

    def filtrarIMG(self, imagen):
      
        lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
        # -----Splitting the LAB image to different channels-------------------------
        l, a, b = cv2.split(lab)

        # -----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        # cv2.imshow('CLAHE output', cl)

        # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl, a, b))

        # -----Converting image from LAB Color model to RGB model--------------------
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        filtrada = self.binarization(final)
        return filtrada

    def detectar(self):
        self.imagen_filtrada=self.filtrarIMG(self.imagen_dedo_segmentado)
        self.imagen_filtrada = cv2.cvtColor(self.imagen_filtrada, cv2.COLOR_BGR2RGB)
        cv2.imshow('pf', self.imagen_filtrada)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        isaCresta=self.verificarCresta()
        if bool(isaCresta):
            self.estatus=True
            return self.estatus,self.imagen_filtrada
        else:
            self.estatus=False
            print('No se detectaron crestas')
            return self.estatus,self.imagen_filtrada
            
            


    