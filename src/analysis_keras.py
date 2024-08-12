import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import csv
import os

class GaitAnalysis:
    def __init__(self, source=0, test=False):
        self.source = source
        self.test = test
        self.cap = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        self.data = []
        self.angular_velocities = []
        self.activities = []
        self.model_velocity = None
        self.model_activity = None

    def run_analysis(self, duration=30):
        if self.test and os.path.exists("gait_data.csv"):
            self.load_data_from_csv("gait_data.csv")
            return
        
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
                        landmarks = results.pose_landmarks.landmark
                        points = self.extract_points(landmarks)

                        if 5 <= (time.time() - start_time) <= 25:
                            self.data.append(points)
                            if len(self.data) > 1:
                                angular_velocity = self.calculate_angular_velocity(self.data[-2], self.data[-1])
                                self.angular_velocities.append(angular_velocity)
                                activity = self.predict_activity(points)
                                self.activities.append(activity)
                    
                    cv2.imshow('Real-time Gait Analysis', image)
                    if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time) > duration:
                        break
        except Exception as e:
            print(f"Error durante el análisis de la marcha: {e}")
        finally:
            self.release_capture()
            self.save_data_to_csv("gait_data.csv")
            self.train_neural_networks()

    def extract_points(self, landmarks):
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

    def calculate_angular_velocity(self, points1, points2):
        angular_velocity = {}
        for key in points1.keys():
            dx = points2[key][0] - points1[key][0]
            dy = points2[key][1] - points1[key][1]
            velocity = np.sqrt(dx**2 + dy**2)
            angular_velocity[key] = velocity
        return angular_velocity

    def predict_activity(self, points):
        if not self.model_activity:
            print("El modelo de actividad no está entrenado.")
            return "desconocido"

        input_data = []
        for key in points.keys():
            input_data.extend([points[key][0], points[key][1]])

        input_data = np.array(input_data).reshape(1, -1)
        prediction = self.model_activity.predict(input_data)
        activity = np.argmax(prediction)

        activity_labels = {0: 'agachado', 1: 'saltando', 2: 'caminando', 3: 'corriendo', 4: 'sentado'}
        return activity_labels.get(activity, "desconocido")

    def prepare_data_for_training(self):
        X = []
        y_velocity = []
        y_activity = []

        for i in range(1, len(self.data)):
            current_frame = []
            for key in self.data[i].keys():
                current_frame.extend([self.data[i][key][0], self.data[i][key][1]])
            X.append(current_frame)
            y_velocity.append(list(self.angular_velocities[i - 1].values()))
            y_activity.append(self.activities[i - 1] if i - 1 < len(self.activities) else 0)

        return np.array(X), np.array(y_velocity), np.array(y_activity)

    def train_neural_networks(self):
        if not self.data or not self.angular_velocities or not self.activities:
            print("No hay datos suficientes para entrenar las redes neuronales.")
            return

        X, y_velocity, y_activity = self.prepare_data_for_training()

        X_train, X_test, y_velocity_train, y_velocity_test = train_test_split(X, y_velocity, test_size=0.2, random_state=42)
        _, _, y_activity_train, y_activity_test = train_test_split(X, y_activity, test_size=0.2, random_state=42)

        # Red neuronal para la predicción de velocidades angulares
        self.model_velocity = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dense(y_velocity_train.shape[1])
        ])

        self.model_velocity.compile(optimizer='adam', loss='mse')
        self.model_velocity.fit(X_train, y_velocity_train, epochs=10, validation_data=(X_test, y_velocity_test))
        test_loss_velocity = self.model_velocity.evaluate(X_test, y_velocity_test)
        print(f"Test loss (velocidad angular): {test_loss_velocity}")

        # Red neuronal para la detección de actividad
        self.model_activity = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dense(5, activation='softmax')  # 5 actividades: agachado, saltando, caminando, corriendo, sentado
        ])

        self.model_activity.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model_activity.fit(X_train, y_activity_train, epochs=10, validation_data=(X_test, y_activity_test))
        test_loss_activity, test_accuracy_activity = self.model_activity.evaluate(X_test, y_activity_test)
        print(f"Test loss (actividad): {test_loss_activity}, Test accuracy (actividad): {test_accuracy_activity}")

    def save_data_to_csv(self, filename):
        if not self.data:
            print("No hay datos disponibles para guardar.")
            return
        
        keys = self.data[0].keys()
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame'] + [f'{key}_x' for key in keys] + [f'{key}_y' for key in keys] + [f'vel_{key}' for key in keys] + ['activity'])
            for i, (points, velocities, activity) in enumerate(zip(self.data, self.angular_velocities, self.activities)):
                row = [i]
                for key in keys:
                    row.extend([points[key][0], points[key][1]])
                for key in keys:
                    row.append(velocities.get(key, ''))
                row.append(activity)
                writer.writerow(row)


    def load_data_from_csv(self, filename):
        self.data = []
        self.angular_velocities = []
        self.activities = []
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                points = {}
                velocities = {}
                activity = 0
                for key in row.keys():
                    if key.endswith('_x'):
                        point_key = key[:-2]
                        points[point_key] = (float(row[key]), float(row[f'{point_key}_y']))
                    elif key.startswith('vel_'):
                        vel_key = key[4:]
                        velocities[vel_key] = float(row[key])
                    elif key == 'activity':
                        activity = int(row[key])
                self.data.append(points)
                self.angular_velocities.append(velocities)
                self.activities.append(activity)

    def release_capture(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    analyzer = GaitAnalysis(source=0, test=False)
    analyzer.run_analysis(duration=30)
