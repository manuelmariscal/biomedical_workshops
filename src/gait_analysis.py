import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import savgol_filter

class GaitAnalysis:
    def __init__(self, source=0):
        """
        Inicializa el análisis de la marcha en tiempo real.
        :param source: Fuente de video (0 para cámara de laptop, URL para cámara de celular).
        """
        self.source = source
        self.cap = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        self.data = []

    def run_analysis(self, duration=30):
        """
        Realiza el análisis de la marcha en tiempo real.
        :param duration: Duración del análisis de la marcha en segundos.
        """
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise ValueError("Error al abrir la fuente de video")
            
            start_time = time.time()
            with self.mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
                while self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Error al capturar el frame")
                        break
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    if results.pose_landmarks:
                        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                        
                        # Extraer las coordenadas de las landmarks
                        landmarks = results.pose_landmarks.landmark
                        points = self.extract_points(landmarks)

                        # Guardar los datos para visualización posterior
                        if 5 <= (time.time() - start_time) <= 25:
                            self.data.append(points)
                    
                    cv2.imshow('Real-time Gait Analysis', image)
                    if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time) > duration:
                        break
        except Exception as e:
            print(f"Error durante el análisis de la marcha: {e}")
        finally:
            self.release_capture()

    def extract_points(self, landmarks):
        """
        Extrae las coordenadas de puntos de interés de las landmarks.
        :param landmarks: Lista de landmarks.
        :return: Diccionario con los puntos de interés y sus coordenadas.
        """
        points = {
            'left_ankle': (landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 1 - landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y),
            'right_ankle': (landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 1 - landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y),
            'left_knee': (landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, 1 - landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y),
            'right_knee': (landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 1 - landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y),
            'left_hip': (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, 1 - landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y),
            'right_hip': (landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x, 1 - landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y),
            'left_shoulder': (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 1 - landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y),
            'right_shoulder': (landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 1 - landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y),
            'left_elbow': (landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 1 - landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y),
            'right_elbow': (landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 1 - landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y),
            'left_wrist': (landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x, 1 - landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y),
            'right_wrist': (landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 1 - landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y),
            'center_of_gravity': (landmarks[self.mp_pose.PoseLandmark.NOSE.value].x, 1 - landmarks[self.mp_pose.PoseLandmark.NOSE.value].y)
        }
        return points

    def save_and_display_movie(self, start_time, end_time):
        """
        Guarda y muestra los datos de la marcha en una animación tipo "movie".
        :param start_time: Tiempo de inicio en segundos.
        :param end_time: Tiempo de fin en segundos.
        """
        # Suavizar los datos
        smoothed_data = self.smooth_data(self.data)
        
        fig, ax = plt.subplots()
        segments = [
            ('left_ankle', 'left_knee'),
            ('left_knee', 'left_hip'),
            ('left_hip', 'center_of_gravity'),
            ('center_of_gravity', 'left_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_ankle', 'right_knee'),
            ('right_knee', 'right_hip'),
            ('right_hip', 'center_of_gravity'),
            ('center_of_gravity', 'right_shoulder'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist')
        ]
        colors = plt.cm.viridis(np.linspace(0, 1, len(segments)))
        lines = [ax.plot([], [], color=color, lw=2)[0] for color in colors]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(frame):
            points = smoothed_data[frame]
            for i, (p1, p2) in enumerate(segments):
                lines[i].set_data([points[p1][0], points[p2][0]], [points[p1][1], points[p2][1]])
            return lines

        # Asegúrate de que los cuadros estén dentro del rango de los datos suavizados
        start_frame = max(0, int(start_time * 30))  # Convertir tiempo a frames (asumiendo 30 FPS)
        end_frame = min(len(smoothed_data), int(end_time * 30))
        ani = FuncAnimation(fig, update, frames=range(start_frame, end_frame), init_func=init, blit=True, interval=33)
        plt.show()

    def smooth_data(self, data):
        """
        Suaviza los datos de las posiciones de los puntos utilizando un filtro de Savitzky-Golay.
        :param data: Lista de diccionarios con las posiciones de los puntos.
        :return: Lista de diccionarios con las posiciones suavizadas de los puntos.
        """
        smoothed_data = []
        keys = data[0].keys()
        for key in keys:
            x = [point[key][0] for point in data]
            y = [point[key][1] for point in data]
            x_smooth = savgol_filter(x, 11, 3)  # Ajustar parámetros según sea necesario
            y_smooth = savgol_filter(y, 11, 3)
            for i, point in enumerate(data):
                if len(smoothed_data) <= i:
                    smoothed_data.append({})
                smoothed_data[i][key] = (x_smooth[i], y_smooth[i])
        return smoothed_data

    def release_capture(self):
        """
        Libera la captura de video.
        """
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
