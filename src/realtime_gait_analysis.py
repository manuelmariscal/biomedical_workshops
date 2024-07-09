import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

class RealTimeGaitAnalysis:
    def __init__(self, source=0):
        """
        Inicializa el análisis de la marcha en tiempo real.
        :param source: Fuente de video (0 para cámara de laptop, URL para cámara de celular).
        """
        self.source = source
        self.cap = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def analyze_gait(self):
        """
        Analiza la marcha en tiempo real.
        """
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise ValueError("Error al abrir la fuente de video")
            
            with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
                        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
                        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
                        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                        
                        # Calcular la distancia entre los tobillos como un ejemplo de análisis
                        ankle_distance = np.sqrt((left_ankle.x - right_ankle.x)**2 + 
                                                 (left_ankle.y - right_ankle.y)**2 + 
                                                 (left_ankle.z - right_ankle.z)**2)
                        print(f'Distancia entre los tobillos: {ankle_distance:.2f}')
                    
                    cv2.imshow('Real-time Gait Analysis', image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        except Exception as e:
            print(f"Error durante el análisis de la marcha: {e}")
        finally:
            self.release_capture()

    def release_capture(self):
        """
        Libera la captura de video.
        """
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
