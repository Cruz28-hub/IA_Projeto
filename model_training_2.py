from ultralytics import YOLO
import os

# --- Configurações do Treino ---
config = {
    "data": os.path.join("super_tux_kart_project_luis-1", "data.yaml"),  # Caminho do seu data.yaml
    "model": "yolov8n.pt",             # Modelo pré-treinado (nano = mais rápido)
    "epochs": 100,                     # Número de épocas
    "imgsz": 620,                      # Tamanho da imagem (reduzido para CPU)
    "batch": 4,                        # Tamanho do batch (ajuste conforme sua RAM)
    "device": "cpu",                   # Usar CPU (mude para "0" se tiver GPU)
    "workers": 2,                      # Núcleos do CPU usado                                  # Ativar augmentations do data.yaml
    "name": "supertuxkart_cpu_train2_sem_argumentaion"   # Nome da pasta de resultados
}

# --- Iniciar Treino ---
print("🚀 Iniciando treino... (Pode levar várias horas em CPU)")
model = YOLO(config["model"])
results = model.train(**config)

print("✅ Treino concluído! Modelo salvo em:", f'runs/detect/{config["name"]}')