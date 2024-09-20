# src/tracking.py

import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os

class Tracking:
    def __init__(self, source=0):
        """
        Inicializa el seguimiento en tiempo real.
        :param source: Fuente de video (0 para cámara de laptop, URL para cámara externa).
        """
        self.source = int(source) if source.isdigit() else source
        self.cap = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        self.data = []
        self.angular_velocities = []
        self.angular_accelerations = []

        self.recording = False
        self.arms_raised_start_time = None
        self.countdown_started = False
        self.countdown_start_time = None
        self.movement_label = None
        self.arm_raised_start_time = None

    def run_analysis(self):
        """
        Realiza el seguimiento en tiempo real.
        """
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise ValueError("Error al abrir la fuente de video")

            with self.mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
                while self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Error al capturar el frame")
                        break

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    height, width, _ = frame.shape

                    if results.pose_landmarks:
                        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                        # Extraer las coordenadas de las landmarks
                        landmarks = results.pose_landmarks.landmark
                        points = self.extract_points(landmarks, width, height)

                        # Detectar si ambos brazos están levantados
                        arms_raised = self.both_arms_raised(landmarks)

                        if arms_raised:
                            if self.arms_raised_start_time is None:
                                self.arms_raised_start_time = time.time()
                            elif time.time() - self.arms_raised_start_time >= 3 and not self.countdown_started:
                                print("Ambos brazos levantados durante 3 segundos. Iniciando conteo regresivo de 2 segundos...")
                                self.countdown_started = True
                                self.countdown_start_time = time.time()
                        else:
                            self.arms_raised_start_time = None
                            self.countdown_started = False
                            self.countdown_start_time = None

                        # Iniciar grabación después del conteo regresivo
                        if self.countdown_started:
                            countdown_elapsed = time.time() - self.countdown_start_time
                            if countdown_elapsed >= 2 and not self.recording:
                                print("Comenzando grabación...")
                                self.recording = True
                                self.data = []
                                self.angular_velocities = []
                                self.angular_accelerations = []
                            else:
                                cv2.putText(image, f"Grabando en {int(2 - countdown_elapsed)}...", (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        # Detectar levantamiento de brazo derecho o izquierdo durante la grabación
                        if self.recording:
                            self.data.append(points)
                            if len(self.data) > 1:
                                angular_velocity = self.calculate_angular_velocity(self.data[-2], self.data[-1])
                                self.angular_velocities.append(angular_velocity)
                            if len(self.angular_velocities) > 1:
                                angular_acceleration = self.calculate_angular_acceleration(
                                    self.angular_velocities[-2], self.angular_velocities[-1])
                                self.angular_accelerations.append(angular_acceleration)

                            movement = self.detect_movement(landmarks)
                            if movement:
                                if self.arm_raised_start_time is None:
                                    self.arm_raised_start_time = time.time()
                                    self.movement_label = movement
                                elif time.time() - self.arm_raised_start_time >= 5:
                                    label = "VALIDO" if movement == "derecho" else "INVALIDO"
                                    filename = f"gait_data_{label}_{int(time.time())}.csv"
                                    print(f"Deteniendo grabación. Movimiento {label}. Guardando datos en {filename}")
                                    self.save_data_to_csv(filename)
                                    self.recording = False
                                    self.countdown_started = False
                                    self.arms_raised_start_time = None
                                    self.arm_raised_start_time = None
                                    self.movement_label = None
                            else:
                                self.arm_raised_start_time = None
                                self.movement_label = None

                            cv2.putText(image, "Grabando...", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # Crear imagen del esqueleto
                        skeleton_image = np.zeros((height, width, 3), dtype=np.uint8)
                        self.draw_skeleton(skeleton_image, points)

                        # Concatenar las imágenes
                        combined_image = cv2.hconcat([image, skeleton_image])

                        # Mostrar la imagen combinada
                        cv2.imshow('Análisis de Movimiento en Tiempo Real', combined_image)
                    else:
                        cv2.imshow('Análisis de Movimiento en Tiempo Real', image)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break

        except KeyboardInterrupt:
            print("\nAnálisis interrumpido por el usuario")
        except Exception as e:
            print(f"Error durante el análisis: {e}")
        finally:
            self.release_capture()

    def extract_points(self, landmarks, image_width, image_height):
        """
        Extrae las coordenadas de puntos de interés de las landmarks en píxeles.
        """
        try:
            points = {
                'left_ankle': (int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x * image_width),
                               int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y * image_height)),
                'right_ankle': (int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].x * image_width),
                                int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y * image_height)),
                'left_knee': (int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x * image_width),
                              int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y * image_height)),
                'right_knee': (int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x * image_width),
                               int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y * image_height)),
                'left_hip': (int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x * image_width),
                             int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y * image_height)),
                'right_hip': (int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x * image_width),
                              int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y * image_height)),
                'left_shoulder': (int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x * image_width),
                                  int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y * image_height)),
                'right_shoulder': (int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * image_width),
                                   int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y * image_height)),
                'left_elbow': (int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x * image_width),
                               int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y * image_height)),
                'right_elbow': (int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x * image_width),
                                int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y * image_height)),
                'left_wrist': (int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x * image_width),
                               int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y * image_height)),
                'right_wrist': (int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x * image_width),
                                int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y * image_height)),
                'center_of_gravity': (int(landmarks[mp.solutions.pose.PoseLandmark.NOSE.value].x * image_width),
                                      int(landmarks[mp.solutions.pose.PoseLandmark.NOSE.value].y * image_height))
            }
            return points
        except Exception as e:
            print(f"Error al extraer puntos: {e}")
            return {}

    def draw_skeleton(self, image, points):
        """
        Dibuja el esqueleto en la imagen dada.
        """
        try:
            connections = [
                ('left_ankle', 'left_knee'),
                ('left_knee', 'left_hip'),
                ('left_hip', 'left_shoulder'),
                ('left_shoulder', 'left_elbow'),
                ('left_elbow', 'left_wrist'),
                ('right_ankle', 'right_knee'),
                ('right_knee', 'right_hip'),
                ('right_hip', 'right_shoulder'),
                ('right_shoulder', 'right_elbow'),
                ('right_elbow', 'right_wrist'),
                ('left_hip', 'right_hip'),
                ('left_shoulder', 'right_shoulder'),
            ]
            # Dibujar los puntos
            for key, (x, y) in points.items():
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            # Dibujar las conexiones
            for (p1, p2) in connections:
                if p1 in points and p2 in points:
                    x1, y1 = points[p1]
                    x2, y2 = points[p2]
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        except Exception as e:
            print(f"Error al dibujar el esqueleto: {e}")

    def both_arms_raised(self, landmarks):
        """
        Detecta si ambos brazos están levantados.
        """
        try:
            left_wrist_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y
            right_wrist_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y
            left_shoulder_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y
            right_shoulder_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y
            if left_wrist_y < left_shoulder_y and right_wrist_y < right_shoulder_y:
                return True
            else:
                return False
        except Exception as e:
            print(f"Error al detectar brazos levantados: {e}")
            return False

    def detect_movement(self, landmarks):
        """
        Detecta si el brazo derecho o izquierdo está levantado durante la grabación.
        """
        try:
            left_wrist_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y
            right_wrist_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y
            left_shoulder_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y
            right_shoulder_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y

            if right_wrist_y < right_shoulder_y and left_wrist_y > left_shoulder_y:
                return "derecho"
            elif left_wrist_y < left_shoulder_y and right_wrist_y > right_shoulder_y:
                return "izquierdo"
            else:
                return None
        except Exception as e:
            print(f"Error al detectar movimiento: {e}")
            return None

    def calculate_angular_velocity(self, points1, points2):
        """
        Calcula la velocidad angular entre dos conjuntos de puntos en 2D.
        """
        angular_velocity = {}
        try:
            for key in points1.keys():
                dx = points2[key][0] - points1[key][0]
                dy = points2[key][1] - points1[key][1]
                velocity = np.sqrt(dx**2 + dy**2)
                angular_velocity[key] = velocity
        except Exception as e:
            print(f"Error al calcular velocidad angular: {e}")
        return angular_velocity

    def calculate_angular_acceleration(self, velocities1, velocities2):
        """
        Calcula la aceleración angular entre dos conjuntos de velocidades angulares.
        """
        angular_acceleration = {}
        try:
            for key in velocities1.keys():
                dv = velocities2[key] - velocities1[key]
                angular_acceleration[key] = dv
        except Exception as e:
            print(f"Error al calcular aceleración angular: {e}")
        return angular_acceleration

    def save_data_to_csv(self, filename):
        """
        Guarda los datos en un archivo CSV.
        """
        if not self.data:
            print("No hay datos para guardar.")
            return

        keys = self.data[0].keys()
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                header = ['frame']
                for key in keys:
                    header.extend([f'{key}_x', f'{key}_y'])
                for key in keys:
                    header.append(f'vel_{key}')
                for key in keys:
                    header.append(f'acc_{key}')
                writer.writerow(header)

                max_length = max(len(self.data), len(self.angular_velocities), len(self.angular_accelerations))
                for i in range(max_length):
                    row = [i]
                    if i < len(self.data):
                        points = self.data[i]
                        for key in keys:
                            row.extend([points[key][0], points[key][1]])
                    else:
                        row.extend([''] * (len(keys) * 2))

                    if i < len(self.angular_velocities):
                        velocities = self.angular_velocities[i]
                        for key in keys:
                            row.append(velocities.get(key, ''))
                    else:
                        row.extend([''] * len(keys))

                    if i < len(self.angular_accelerations):
                        accelerations = self.angular_accelerations[i]
                        for key in keys:
                            row.append(accelerations.get(key, ''))
                    else:
                        row.extend([''] * len(keys))

                    writer.writerow(row)
        except Exception as e:
            print(f"Error al guardar datos en CSV: {e}")

    def release_capture(self):
        """
        Libera la captura de video.
        """
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
