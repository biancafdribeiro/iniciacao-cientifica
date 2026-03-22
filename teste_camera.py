import cv2
import mediapipe as mp

# Inicializando as ferramentas de desenho e os modelos
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic  # Rosto + mãos
mp_pose = mp.solutions.pose          # Corpo (melhorado)

# 1. Ligando a sua Webcam (CAP_DSHOW ajuda no Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Melhorando a resolução da câmera (ajuda no tracking)
cap.set(3, 1280)
cap.set(4, 720)

# Verificando se a câmera abriu corretamente
if not cap.isOpened():
    print("Erro ao abrir câmera")
    exit()

# 2. Configurando os modelos:
# - Holistic: rosto + mãos
# - Pose: corpo com mais estabilidade
with mp_holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as holistic, mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as pose:

    print("Câmera ligada! Pressione a tecla 'q' para sair.")

    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            print("Ignorando frame vazio da câmera.")
            continue

        # O OpenCV lê a imagem em BGR, mas o MediaPipe precisa de RGB. Fazemos a conversão:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Para melhorar a performance, marcamos a imagem como não gravável temporariamente
        frame_rgb.flags.writeable = False

        # 3. Processamento:
        # - Holistic: rosto + mãos
        # - Pose: corpo (mais preciso)
        resultados_holistic = holistic.process(frame_rgb)
        resultados_pose = pose.process(frame_rgb)

        # Voltamos a imagem para BGR para o OpenCV conseguir desenhar na tela
        frame_rgb.flags.writeable = True
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # =========================
        # 4. DESENHANDO OS PONTOS
        # =========================

        # 🔥 CORPO (usando Pose - melhor qualidade)
        if resultados_pose.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr,
                resultados_pose.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=3, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )

        # 😀 ROSTO (usando Holistic)
        if resultados_holistic.face_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr,
                resultados_holistic.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1)
            )

        # ✋ MÃO ESQUERDA
        if resultados_holistic.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr,
                resultados_holistic.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2)
            )

        # ✋ MÃO DIREITA
        if resultados_holistic.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr,
                resultados_holistic.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2)
            )

        # 5. Mostrando a imagem na tela do seu computador
        cv2.imshow('Reconhecimento de Landmarks - IC (Melhorado)', frame_bgr)

        # Condição de parada: se apertar a letra 'q' no teclado, o programa fecha
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Fechando a câmera e as janelas quando terminar
cap.release()
cv2.destroyAllWindows()