from ultralytics import YOLO

# Load a model
model = YOLO('/media/jeffersonc/DiscoRespaldo/YOLOv8/DeteccionCedulas/Det_Cedulas_Anv_Rev.pt')  # load a custom trained model

# Export the model
model.export(format='tfjs') 