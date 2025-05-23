# 🏎️ SuperTuxKart YOLOv8 Bot

Este script utiliza **YOLOv8** para detetar elementos do jogo SuperTuxKart em tempo real, como o carro do jogador, estrada, obstáculos e itens. Com base nas deteções, o script simula teclas para controlar automaticamente o kart durante a corrida.

## 📦 Requisitos

Antes de correr o script, certifique-se de que tens o seguinte instalado:

- Python 3.8 ou superior
- [YOLOv8 (Ultralytics)](https://docs.ultralytics.com/)
- OpenCV (`opencv-python`)
- PyAutoGUI (`pyautogui`)
- MSS (`mss`)
- NumPy

Pode instalar as dependências com:

```bash
pip install ultralytics opencv-python pyautogui mss numpy
```

⚠️ **Nota:** Certifique-se de que o jogo está em ecrã completo no monitor principal. O script captura o ecrã e simula teclas de forma contínua.

## 📂 Estrutura

O script faz o seguinte:

- Captura o ecrã com `mss`.
- Deteta objetos relevantes usando um modelo YOLOv8 (treinado previamente).
- Calcula direções de movimento com base na posição da estrada e obstáculos.
- Usa `pyautogui` para simular as teclas:
  - `← / →` para virar,
  - `↑` para acelerar,
  - `N` para usar nitro (quando disponível).

## 🚀 Como Correr

2. **Atualiza o caminho para o modelo no script:**

```python
model = YOLO("C:/path/to/model/weights/best.pt")
```

3. **Corre o script:**

```bash
python real_time_inference.py
```

4. Abre o SuperTuxKart em ecrã completo e entra numa corrida. O bot irá controlar automaticamente o kart com base nas deteções.

## 🛑 Parar a Execução

Para interromper o script, pressiona `Ctrl + C` na consola.