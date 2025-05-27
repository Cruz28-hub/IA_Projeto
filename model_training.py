from IPython import display
display.clear_output()

import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dotenv import load_dotenv


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
load_dotenv()  # Carrega as variáveis do .env
api_key = os.getenv("ROBOFLOW_API_KEY")  # Lê a chave da API do ambiente

rf = roboflow.Roboflow(api_key)

# substituir nome do workspace e do projeto
project = rf.workspace("supertuxkartluis").project("super_tux_kart_project_luis")
# se versão do dataset > 1, substituir pela versão correspondente
dataset = project.version(8).download("yolov8")
# WARN: necessário verificar os paths no ficheiro data.yaml, após este ser descarregado

model = YOLO("yolov8s.pt")  # carregar o modelo pre-treinado que se descarregou

# Treinar o modelo
results = model.train(data='super_tux_kart_project_luis-8/data.yaml', epochs=50, imgsz=640, device='cpu',
                       project='runs/version_8_50Epochs',  # intel/window
    name='treino_versao_8_50Epochs',)  # intel/window