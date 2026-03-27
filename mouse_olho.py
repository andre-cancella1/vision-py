import cv2
import mediapipe as mp
import pyautogui
import os
import numpy as np

# --- Configurações do PyAutoGUI ---
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
screen_width, screen_height = pyautogui.size()

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'face_landmarker.task'

if not os.path.exists(model_path):
    print(f"ERRO: Arquivo {model_path} não encontrado.")
    exit()

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True, # Nome corrigido aqui
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

# Variáveis para suavização do mouse (para não tremer)
suave_x, suave_y = 0, 0
fator_suavizacao = 0.2 

print("Script iniciado. Pressione 'q' para sair.")

while True:
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)

    # Verifica se detectou algum rosto
    if detection_result.face_landmarks:
        # Pega a lista de pontos da primeira face
        face = detection_result.face_landmarks[0]

        # --- 1. MOVIMENTO (ÍRIS) ---
        iris_ponto = face[468] # Centro da íris
        
        # Mapeia o movimento sutil do olho para a tela inteira
        alvo_x = np.interp(iris_ponto.x, (0.43, 0.57), (0, screen_width))
        alvo_y = np.interp(iris_ponto.y, (0.43, 0.57), (0, screen_height))
        
        # Aplica suavização
        suave_x = (suave_x * (1 - fator_suavizacao)) + (alvo_x * fator_suavizacao)
        suave_y = (suave_y * (1 - fator_suavizacao)) + (alvo_y * fator_suavizacao)
        
        pyautogui.moveTo(suave_x, suave_y)

        # --- 2. FEEDBACK VISUAL ---
        ix = int(iris_ponto.x * w)
        iy = int(iris_ponto.y * h)
        cv2.circle(frame, (ix, iy), 5, (0, 255, 0), -1)

        # --- 3. CLIQUE (PISCADA) ---
        palpebra_sup = face[159]
        palpebra_inf = face[145]
        distancia_vertical = (palpebra_inf.y - palpebra_sup.y) * h

# Se a abertura for menor que 5.0, o clique acontece.
# Como seu 'aberto' mais baixo foi 5.26, o 5.0 deve dar segurança.
        if distancia_vertical < 5.0: 
            cv2.putText(frame, "CLIQUE!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            pyautogui.click()
            pyautogui.sleep(0.4)

    cv2.imshow('Mouse por Olhar', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()