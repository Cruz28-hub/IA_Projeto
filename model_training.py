from IPython import display
display.clear_output()

import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import roboflow

import ultralytics
from ultralytics import YOLO
ultralytics.checks()

# descarregar o dataset do roboflow, depois de etiquetadas as imagens e criado o dataset
# em Mac tive problemas de permissões e foi necessário dar permissões à pasta de conf. do roboflow:
# - sudo mkdir /Users/davidecarneiro/.config/roboflow (criar pasta onde vai guardar a conf.)
# - sudo chown -R davidecarneiro:staff ~/.config/roboflow (dar permissões ao meu user)
roboflow.login()

# criar um ficheiro com a key da api do roboflow, ou simplesmente substituir abaixo
with open('api_key', "r") as file:
    api_key = file.read().strip() 

rf = roboflow.Roboflow(api_key)

# substituir nome do workspace e do projeto
project = rf.workspace("estg-e1qbq").project("supertuxkart")
# se versão do dataset > 1, substituir pela versão correspondente
dataset = project.version(7).download("yolov8")
# WARN: necessário verificar os paths no ficheiro data.yaml, após este ser descarregado

# treinar o modelo
# lista de modelos pre-treinados disponível em https://docs.ultralytics.com/models/yolov8/#performance-metrics
model = YOLO("yolov8s.pt")  # carregar o modelo pre-treinado que se descarregou

# Treinar o modelo
results = model.train(data='SuperTuxKart-7/data.yaml', epochs=100, imgsz=640, device='cpu')  # intel/window