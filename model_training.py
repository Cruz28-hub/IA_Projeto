from IPython import display

display.clear_output()

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import roboflow

import ultralytics
from ultralytics import YOLO

ultralytics.checks()

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
WORKSPACE_NAME = os.getenv('WORKSPACE_NAME')
PROJECT_NAME = os.getenv('PROJECT_NAME')

# descarregar o dataset do roboflow, depois de etiquetadas as imagens e criado o dataset
# em Mac tive problemas de permissões e foi necessário dar permissões à pasta de conf. do roboflow:
# - sudo mkdir /Users/davidecarneiro/.config/roboflow (criar pasta onde vai guardar a conf.)
# - sudo chown -R davidecarneiro:staff ~/.config/roboflow (dar permissões ao meu user)
roboflow.login

# criar um ficheiro com a key da api do roboflow, ou simplesmente substituir abaixo
# with open('api_key', "r") as file:
#     api_key = file.read().strip()

rf = roboflow.Roboflow(API_KEY)

# substituir nome do workspace e do projeto
project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
# se versão do dataset > 1, substituir pela versão correspondente
dataset = project.version(1).download("yolov8")
# WARN: necessário verificar os paths no ficheiro data.yaml, após este ser descarregado
# lista de modelos pre-treinados disponível em https://docs.ultralytics.com/models/yolov8/#performance-metrics
model = YOLO("yolov8m.pt")  # carregar o modelo pre-treinado que se descarregou

# Treinar o modelo
results = model.train(data='./8200690_OD_V2-1/data.yaml', epochs=20, imgsz=640, device='cpu')  # intel/window
image = "./8200690_OD_V2-1/train/images"
results = model.predict(
    source=image,
    conf=0.05,  # Valor padrão
    classes=[0]  # Apenas Track-c6Id se necessário
)
