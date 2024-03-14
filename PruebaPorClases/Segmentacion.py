from ultralytics import YOLO
from PIL import Image
import os
import errno
import cv2
import numpy as np


class Segmentacion:
    
    def loadModel(self):
        directorio_actual = os.path.dirname(__file__) 
        path_model = os.path.join(directorio_actual, 'YOLOv8Segmentation50epoch.pt')
        model = YOLO(path_model)
        return model

    def __init__(self, imagen):
        self.imagen_dedo = imagen
        self.imagen_dedo_segmentada = None
        self.status=None
        #self.model_segmetation = self.__class__.loadModel()  # Accede al método de clase
        self.model_segmetation = self.loadModel()  # Llama a loadModel() como un método de instancia
        current_dir = os.getcwd()
        print(current_dir)


    def segmentarDedo(self):
        results = self.model_segmetation(self.imagen_dedo,imgsz=640) 
        return results
    
    def verifyFingerInIMage(self):
        flag=None
        isaFinger=False
        resultados=self.segmentarDedo()
        for r in resultados:
            flag=r.boxes
        if bool(flag):
            isaFinger=True
        else:
            isaFinger=False
            print('No se detecto Dedo')
        return isaFinger,resultados
    
    def recorteRectangular(self,numpyArrayCOrd,imageToCrop):
        x1=numpyArrayCOrd[0][0]
        y1=numpyArrayCOrd[0][1]
        x2=numpyArrayCOrd[0][2]
        y2=numpyArrayCOrd[0][3]
        cropped_region = imageToCrop[int(y1):int(y2), int(x1):int(x2)]
        # Crear una máscara para identificar píxeles negros
        mask = (cropped_region == [0, 0, 0]).all(axis=2)

        # Cambiar el fondo negro a blanco
        cropped_region[mask] = [255, 255, 255]

        return cropped_region
    
    def recorteContorno(self,imagen,path_to_txt,coordenadasRecorte):

        # Ruta del archivo .txt con las coordenadas
        txt_path = path_to_txt

        # Cargar la imagen original
        image = imagen

        # Inicializar una máscara vacía
        mask = np.zeros_like(image)

        # Cargar las coordenadas desde el archivo .txt
        with open(txt_path, 'r') as file:
            lines = file.readlines()

        # Iterar a través de las líneas del archivo
        for line in lines:
            # Dividir la línea en partes y omitir la clase (primer número)
            parts = line.strip().split()[1:]
            
            # Convertir las partes en una lista de puntos (x, y)
            points = [(int((float(parts[i])+0.01) * image.shape[1]), int((float(parts[i+1])-0.01) * image.shape[0])) for i in range(0, len(parts), 2)]
            
            # Convertir la lista de puntos en un formato compatible con OpenCV
            points = np.array(points, dtype=np.int32)
            
            # Dibujar el polígono en la máscara
            cv2.fillPoly(mask, [points], (255, 255, 255))

            # Recortar la región de interés de la imagen original usando la máscara
            result = cv2.bitwise_and(image, mask)
            result=self.recorteRectangular(coordenadasRecorte,result)
            #output_image=os.path.join(outhput_path,f'{name}segmented3.png')
            #cv2.imwrite(output_image, result)
            return result

    def recortarDedoLocalizado(self):
        verify,results=self.verifyFingerInIMage()
        if bool(verify):
            for r in results:
                path_to_txt=os.path.join(os.path.dirname(__file__),'Nueva carpeta','coordenadas.txt')
                r.save_txt(path_to_txt)
                box=r.boxes
                box=box.xyxy
                coord=box.numpy()
                self.imagen_dedo_segmentada=self.recorteContorno(self.imagen_dedo,path_to_txt,coord)
            self.status=True 
            return self.imagen_dedo_segmentada,self.status
        else:
            self.status=False
            return self.imagen_dedo_segmentada,self.status


    