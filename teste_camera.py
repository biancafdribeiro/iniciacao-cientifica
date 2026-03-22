import cv2 # OpenCV 
import mediapipe as mp # Landmarks
import csv # Biblioteca nova para criar a planilha Excel!
import time # Mede o tempo de execução

# =========================
# CONFIGURAÇÃO DA PLANILHA
# =========================
arquivo_csv = 'dados_experimento.csv'
cabecalho = ['Frame', 'Tempo', 'Nariz_X', 'Nariz_Y', 'OmbroEsq_X', 'OmbroEsq_Y', 'OmbroDir_X', 'OmbroDir_Y']

# Criando o arquivo e escrevendo a primeira linha (cabeçalho)
with open(arquivo_csv, mode='w', newline='') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(cabecalho)

# Inicializando as ferramentas de desenho e os modelos
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic   
mp_pose = mp.solutions.pose  # Somente corpo       

# Ligando a sua Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

if not cap.isOpened():
    print("Erro ao abrir câmera")
    exit()

frame_count = 0 # Contador de frames para a planilha
tempo_inicial = time.time() # Cronômetro

with mp_holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as holistic, mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as pose:

    print("Câmera ligada! Gravando dados na planilha... Pressione 'q' para sair.")

    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            continue

        frame_count += 1
        tempo_atual = round(time.time() - tempo_inicial, 2) # Tempo em segundos

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        resultados_holistic = holistic.process(frame_rgb)
        resultados_pose = pose.process(frame_rgb)

        frame_rgb.flags.writeable = True
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # =========================
        # EXTRAÇÃO DE DADOS (A CIÊNCIA)
        # =========================
        if resultados_pose.pose_landmarks:
            # Pegando a lista de todos os pontos do corpo
            landmarks = resultados_pose.pose_landmarks.landmark

            # Extraindo as coordenadas (X e Y) dos pontos que nos interessam
            # O .x e .y retornam um valor de 0.0 a 1.0 (onde a tela está)
            nariz_x = landmarks[mp_pose.PoseLandmark.NOSE.value].x
            nariz_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
            
            ombro_esq_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
            ombro_esq_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            
            ombro_dir_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
            ombro_dir_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y

            # Salvando os números na planilha automaticamente
            with open(arquivo_csv, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([frame_count, tempo_atual, nariz_x, nariz_y, ombro_esq_x, ombro_esq_y, ombro_dir_x, ombro_dir_y])

            # DESENHANDO O CORPO NA TELA
            mp_drawing.draw_landmarks(
                frame_bgr, resultados_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=3, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )

        # DESENHANDO O ROSTO
        if resultados_holistic.face_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr, resultados_holistic.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1)
            )

        # DESENHANDO AS MÃOS
        if resultados_holistic.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame_bgr, resultados_holistic.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if resultados_holistic.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame_bgr, resultados_holistic.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Mostrando a tela
        cv2.imshow('Coletor de Dados IC - Lacuna 3', frame_bgr)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Coleta finalizada! Verifique o arquivo '{arquivo_csv}' na sua pasta.")