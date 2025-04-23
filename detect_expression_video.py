import cv2
from deepface import DeepFace
import mediapipe as mp
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Constants for DeepFace detector backends
DETECTOR_BACKEND_OPENCV = 'opencv'
DETECTOR_BACKEND_SSD = 'ssd'
DETECTOR_BACKEND_DLIB = 'dlib'
DETECTOR_BACKEND_MTCNN = 'mtcnn'
DETECTOR_BACKEND_RETINAFACE = 'retinaface'
DETECTOR_BACKEND_MEDIAPIPE = 'mediapipe'

# Default detector backend
DEFAULT_DETECTOR_BACKEND = DETECTOR_BACKEND_MTCNN

def detect_pose_action(pose_landmarks):
    """
    Detects various pose actions like arm up, leg up, squat, jump, etc.

    Args:
        pose_landmarks (list): List of (x, y, z) tuples for pose landmarks.

    Returns:
        dict: Actions detected (e.g., 'arm_up': True, 'leg_up': False, 'squat': False).
    """
    actions = {
        'arm_up': False,
        'leg_up': False,
        'both_arms_up': False,
        'squat': False,
        'jump': False,
        'head_tilt_left': False,
        'head_tilt_right': False
    }

    if pose_landmarks:
        # Arm up detection
        left_wrist = pose_landmarks[15]  # Left wrist
        left_shoulder = pose_landmarks[11]  # Left shoulder
        right_wrist = pose_landmarks[16]  # Right wrist
        right_shoulder = pose_landmarks[12]  # Right shoulder

        if left_wrist[1] < left_shoulder[1]:
            actions['arm_up'] = True
        if right_wrist[1] < right_shoulder[1]:
            actions['arm_up'] = True
        if left_wrist[1] < left_shoulder[1] and right_wrist[1] < right_shoulder[1]:
            actions['both_arms_up'] = True

        # Leg up detection
        left_ankle = pose_landmarks[27]  # Left ankle
        left_hip = pose_landmarks[23]  # Left hip
        right_ankle = pose_landmarks[28]  # Right ankle
        right_hip = pose_landmarks[24]  # Right hip

        if left_ankle[1] < left_hip[1]:
            actions['leg_up'] = True
        if right_ankle[1] < right_hip[1]:
            actions['leg_up'] = True

        # Squat detection
        left_knee = pose_landmarks[25]  # Left knee
        right_knee = pose_landmarks[26]  # Right knee
        if left_hip[1] > left_knee[1] and right_hip[1] > right_knee[1]:
            actions['squat'] = True

        # Jump detection
        if left_ankle[1] < 0.5 and right_ankle[1] < 0.5:  # Assuming normalized coordinates
            actions['jump'] = True

        # Head tilt detection
        head = pose_landmarks[0]  # Head (nose)
        left_ear = pose_landmarks[7]  # Left ear
        right_ear = pose_landmarks[8]  # Right ear

        if head[0] < left_ear[0]:  # Head tilted to the left
            actions['head_tilt_left'] = True
        if head[0] > right_ear[0]:  # Head tilted to the right
            actions['head_tilt_right'] = True

    return actions

def detect_anomalous_movement(pose_landmarks, prev_pose_landmarks, threshold=0.2):
    """
    Detecta movimentos anômalos com base na variação dos landmarks entre frames consecutivos.

    Args:
        pose_landmarks (list): Lista de (x, y, z) dos landmarks atuais.
        prev_pose_landmarks (list): Lista de (x, y, z) dos landmarks do frame anterior.
        threshold (float): Limite para detectar movimentos bruscos.

    Returns:
        bool: True se o movimento for anômalo, False caso contrário.
    """
    if not prev_pose_landmarks or not pose_landmarks:
        return False  # Não há dados suficientes para comparação

    total_variation = 0
    for current, previous in zip(pose_landmarks, prev_pose_landmarks):
        variation = np.linalg.norm(np.array(current) - np.array(previous))
        total_variation += variation

    # Média da variação por landmark
    avg_variation = total_variation / len(pose_landmarks)
    return avg_variation > threshold

def detect_emotions(video_path, output_path):
    # Capturar vídeo do arquivo especificado
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(video_path)

    # Verificar se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obter propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # Variáveis para estatísticas
    frame_count = 0
    anomaly_count = 0
    emotion_counts = {}
    action_counts = {}
    prev_pose_landmarks = None  # Inicializar landmarks do frame anterior
    # Loop para processar cada frame do vídeo
    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        anomalies_in_frame = False

        try:
            # Analisar o frame para detectar faces e expressões
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True, align=True,detector_backend=DEFAULT_DETECTOR_BACKEND)

            # Iterar sobre cada face detectada
            for face in result:
                # Obter a caixa delimitadora da face
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']

                # Obter a emoção dominante
                dominant_emotion = face['dominant_emotion']

                # Desenhar um retângulo ao redor da face
                # Calculate the new dimensions for the rectangle (half the size)
                new_w, new_h = w // 2, h // 2
                new_x, new_y = x + w // 4, y + h // 4

                # Draw a smaller rectangle around the face
                cv2.rectangle(frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)

                # Write the dominant emotion above the smaller rectangle
                text_x, text_y = new_x, new_y - 10  # Position the text slightly above the rectangle
                cv2.putText(frame, dominant_emotion, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                emotion_counts[dominant_emotion] = emotion_counts.get(dominant_emotion, 0) + 1
                anomalies_in_frame = False
        except ValueError as e:
            pass

        # Escrever o frame processado no vídeo de saída
        out.write(frame)

        # Converter o frame para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processar o frame para detectar a pose
        results = pose.process(rgb_frame)

        # Desenhar as anotações da pose no frame
        if results.pose_landmarks:
            pose_landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            actions = detect_pose_action(pose_landmarks)
            for action, detected in actions.items():
                if detected:
                    action_counts[action] = action_counts.get(action, 0) + 1

            # Detectar movimentos anômalos
            if detect_anomalous_movement(pose_landmarks, prev_pose_landmarks):
                anomalies_in_frame = True

            prev_pose_landmarks = pose_landmarks  # Atualizar landmarks do frame anterior
            # Escrever as ações detectadas no frame
            action_text = ', '.join([action for action, detected in actions.items() if detected])
            cv2.putText(frame, action_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Desenhar as anotações da pose no frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if anomalies_in_frame:
            anomaly_count += 1
            cv2.putText(frame, "Movimento Anômalo", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        # Escrever o frame processado no vídeo de saída
        out.write(frame)

    # Liberar a captura de vídeo e fechar todas as janelas
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Exibir estatísticas
    print("Estatísticas do Vídeo:")
    print(f"1. **Frames Analisados**: {frame_count}")
    print(f"2. **Anomalias Detectadas**: {anomaly_count}")
    print(f"3. **Emoções Principais**: {emotion_counts}")
    print(f"4. **Atividades Principais**: {action_counts}")

# Caminho para o arquivo de vídeo na mesma pasta do script
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'input_video.mp4')  # Substitua 'meu_video.mp4' pelo nome do seu vídeo
output_video_path = os.path.join(script_dir, f'output_video_{DEFAULT_DETECTOR_BACKEND}.mp4')  # Nome do vídeo de saída


# Chamar a função para detectar emoções no vídeo e salvar o vídeo processado
detect_emotions(input_video_path, output_video_path)
