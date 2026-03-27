import cv2
import mediapipe as mp
import pyautogui
import os
import numpy as np

# --- Configurações do PyAutoGUI (Essencial para suavidade) ---
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
screen_width, screen_height = pyautogui.size()

# --- Configurações do Novo MediaPipe Tasks ---
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Caminho para o modelo que você baixou
model_path = 'hand_landmarker.task'

if not os.path.exists(model_path):
    print(f"ERRO: Arquivo {model_path} não encontrado na pasta.")
    print("Baixe em: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    exit()

# Inicializa o detector
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# --- Configurações da Câmera ---
cap = cv2.VideoCapture(0)
index_x, index_y = 0, 0
thumb_x, thumb_y = 0, 0

print("Script iniciado. Pressione 'q' para sair.")

while True:
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]

        idx_landmark = hand_landmarks[8]
        idx_x_px = int(idx_landmark.x * w)
        idx_y_px = int(idx_landmark.y * h)
        
        thb_landmark = hand_landmarks[4]
        thb_x_px = int(thb_landmark.x * w)
        thb_y_px = int(thb_landmark.y * h)

        cv2.circle(frame, (idx_x_px, idx_y_px), 10, (0, 255, 0), cv2.FILLED) # Verde: Indicador
        cv2.circle(frame, (thb_x_px, thb_y_px), 10, (255, 0, 0), cv2.FILLED) # Azul: Polegar

        move_x = np.interp(idx_x_px, (70, w - 70), (0, screen_width))
        move_y = np.interp(idx_y_px, (70, h - 70), (0, screen_height))
        
        pyautogui.moveTo(move_x, move_y)

        distancia = ((idx_x_px - thb_x_px)**2 + (idx_y_px - thb_y_px)**2)**0.5
        
        if distancia < 30: 
            cv2.circle(frame, (idx_x_px, idx_y_px), 15, (0, 0, 255), cv2.FILLED) # Vermelho: Clicando
            pyautogui.click()
            pyautogui.sleep(0.2) # Pequeno delay para evitar cliques triplos

    cv2.imshow('Mouse Virtual tasks', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()