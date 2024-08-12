import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import savgol_filter
import csv
import os
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

class GaitAnalysis:
    def __init__(self, source=0, test=False):
        """
        Inicializa el análisis de la marcha en tiempo real.
        :param source: Fuente de video (0 para cámara de laptop, URL para cámara de celular).
        :param test: Indica si se debe usar un archivo CSV guardado previamente para el análisis.
        """
        self.source = source
        self.test = test
        self.cap = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.tracker = cv2.TrackerKCF_create()
        self.tracking = False
        self.bbox = None

        self.data = []
        self.angular_velocities = []
        self.angular_accelerations = []
        self.activities = []
        self.model_activity = None

    def run_analysis(self, duration=30):
        """
        Realiza el análisis de la marcha en tiempo real.
        :param duration: Duración del análisis de la marcha en segundos.
        """
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

                    # Aplicar sustracción de fondo para detectar movimiento
                    fgmask = self.fgbg.apply(frame)
                    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, None)
                    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, None)

                    # Procesar la imagen con MediaPipe para detección de pose
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)
                    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                    if results.pose_landmarks:
                        self.mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                        landmarks = results.pose_landmarks.landmark
                        points = self.extract_points(landmarks)

                        # Guardar los datos para visualización posterior
                        if 5 <= (time.time() - start_time) <= 25:
                            self.data.append(points)
                            if len(self.data) > 1:
                                angular_velocity = self.calculate_angular_velocity(self.data[-2], self.data[-1])
                                self.angular_velocities.append(angular_velocity)
                            if len(self.angular_velocities) > 1:
                                angular_acceleration = self.calculate_angular_acceleration(self.angular_velocities[-2], self.angular_velocities[-1])
                                self.angular_accelerations.append(angular_acceleration)

                            activity = self.predict_activity(points)
                            self.activities.append(activity)

                    # Inicializar o actualizar el seguimiento
                    if not self.tracking:
                        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            if cv2.contourArea(contour) > 1000:
                                x, y, w, h = cv2.boundingRect(contour)
                                self.bbox = (x, y, w, h)
                                self.tracker.init(frame, self.bbox)
                                self.tracking = True
                    else:
                        self.tracking, self.bbox = self.tracker.update(frame)
                        if self.tracking:
                            x, y, w, h = map(int, self.bbox)
                            cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    cv2.imshow('Real-time Gait Analysis', image_bgr)
                    if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time) > duration:
                        break
        except Exception as e:
            print(f"Error durante el análisis de la marcha: {e}")
        finally:
            self.release_capture()
            self.save_data_to_csv("gait_data.csv")
            self.train_activity_model()

    def extract_points(self, landmarks):
        """
        Extrae las coordenadas de puntos de interés de las landmarks en 2D.
        :param landmarks: Lista de landmarks.
        :return: Diccionario con los puntos de interés y sus coordenadas (x, y).
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

    def calculate_angular_velocity(self, points1, points2):
        """
        Calcula la velocidad angular entre dos conjuntos de puntos en 2D.
        :param points1: Primer conjunto de puntos.
        :param points2: Segundo conjunto de puntos.
        :return: Diccionario con las velocidades angulares.
        """
        angular_velocity = {}
        for key in points1.keys():
            dx = points2[key][0] - points1[key][0]
            dy = points2[key][1] - points1[key][1]
            velocity = np.sqrt(dx**2 + dy**2)
            angular_velocity[key] = velocity
        return angular_velocity

    def calculate_angular_acceleration(self, velocities1, velocities2):
        """
        Calcula la aceleración angular entre dos conjuntos de velocidades angulares.
        :param velocities1: Primer conjunto de velocidades angulares.
        :param velocities2: Segundo conjunto de velocidades angulares.
        :return: Diccionario con las aceleraciones angulares.
        """
        angular_acceleration = {}
        for key in velocities1.keys():
            dv = velocities2[key] - velocities1[key]
            angular_acceleration[key] = dv
        return angular_acceleration

    def predict_activity(self, points):
        """
        Predice la actividad realizada basada en las posiciones articulares.
        :param points: Diccionario con las posiciones de las articulaciones.
        :return: Nombre de la actividad predicha.
        """
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
        y_activity = []

        for i in range(1, len(self.data)):
            current_frame = []
            for key in self.data[i].keys():
                current_frame.extend([self.data[i][key][0], self.data[i][key][1]])
            X.append(current_frame)
            y_activity.append(self.activities[i - 1] if i - 1 < len(self.activities) else 0)

        return np.array(X), np.array(y_activity)

    def train_activity_model(self):
        if not self.data or not self.activities:
            print("No hay datos suficientes para entrenar el modelo de actividad.")
            return

        X, y_activity = self.prepare_data_for_training()

        X_train, X_test, y_activity_train, y_activity_test = train_test_split(X, y_activity, test_size=0.2, random_state=42)

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

    def save_and_display_movie(self, start_time, end_time):
        """
        Guarda y muestra los datos de la marcha en una animación tipo "movie".
        :param start_time: Tiempo de inicio en segundos.
        :param end_time: Tiempo de fin en segundos.
        """
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
        center_of_gravity_line, = ax.plot([], [], 'r-', alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Ancho')
        ax.set_ylabel('Alto')

        def init():
            for line in lines:
                line.set_data([], [])
            center_of_gravity_line.set_data([], [])
            return lines + [center_of_gravity_line]

        def update(frame):
            points = smoothed_data[frame]
            for i, (p1, p2) in enumerate(segments):
                lines[i].set_data([points[p1][0], points[p2][0]], [points[p1][1], points[p2][1]])
            x_cog = [pt['center_of_gravity'][0] for pt in smoothed_data[:frame + 1]]
            y_cog = [pt['center_of_gravity'][1] for pt in smoothed_data[:frame + 1]]
            center_of_gravity_line.set_data(x_cog, y_cog)
            if frame < len(self.angular_velocities):
                velocities = self.angular_velocities[frame]
                accelerations = self.angular_accelerations[frame - 1] if frame > 0 else {}
                velocity_str = '\n'.join([f'{key}: {abs(value):.2f} m/s' for key, value in velocities.items()])
                acceleration_str = '\n'.join([f'{key}: {abs(value):.2f} m/s²' for key, value in accelerations.items()])
                ax.set_title(f'Velocidades:\n{velocity_str}\n\nAceleraciones:\n{acceleration_str}', fontsize=10)
            return lines + [center_of_gravity_line]

        start_frame = max(0, int(start_time * 30))
        end_frame = min(len(smoothed_data), int(end_time * 30))
        ani = FuncAnimation(fig, update, frames=range(start_frame, end_frame), init_func=init, blit=True, interval=33)

        fig2, ax2 = plt.subplots()
        keypoints = [
            'left_ankle', 'right_ankle', 'left_knee', 'right_knee',
            'left_hip', 'right_hip', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
        ]
        trails = {kp: ax2.plot([], [], 'o-', alpha=0.3)[0] for kp in keypoints}
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Ancho')
        ax2.set_ylabel('Alto')

        def init_trails():
            for trail in trails.values():
                trail.set_data([], [])
            return list(trails.values())

        def update_trails(frame):
            points = smoothed_data[frame]
            for kp in keypoints:
                x_kp = [pt[kp][0] for pt in smoothed_data[:frame + 1]]
                y_kp = [pt[kp][1] for pt in smoothed_data[:frame + 1]]
                trails[kp].set_data(x_kp, y_kp)
            return list(trails.values())

        ani2 = FuncAnimation(fig2, update_trails, frames=range(start_frame, end_frame), init_func=init_trails, blit=True, interval=33)

        fig3, ax3 = plt.subplots()
        fading_trails = {kp: ax3.plot([], [], 'o-', alpha=0.3)[0] for kp in keypoints}
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_xlabel('Ancho')
        ax3.set_ylabel('Alto')

        def init_fading_trails():
            for trail in fading_trails.values():
                trail.set_data([], [])
            return list(fading_trails.values())

        def update_fading_trails(frame):
            points = smoothed_data[frame]
            for kp in keypoints:
                x_kp = [pt[kp][0] for pt in smoothed_data[max(0, frame - 10):frame + 1]]
                y_kp = [pt[kp][1] for pt in smoothed_data[max(0, frame - 10):frame + 1]]
                fading_trails[kp].set_data(x_kp, y_kp)
                alpha_values = np.linspace(0.1, 1, len(x_kp))
                for i in range(len(x_kp) - 1):
                    ax3.plot(x_kp[i:i + 2], y_kp[i:i + 2], 'o-', alpha=alpha_values[i])
            return list(fading_trails.values())

        ani3 = FuncAnimation(fig3, update_fading_trails, frames=range(start_frame, end_frame), init_func=init_fading_trails, blit=True, interval=33)

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
            x_smooth = savgol_filter(x, 11, 3)
            y_smooth = savgol_filter(y, 11, 3)
            for i, point in enumerate(data):
                if len(smoothed_data) <= i:
                    smoothed_data.append({})
                smoothed_data[i][key] = (x_smooth[i], y_smooth[i])
        return smoothed_data

    def save_data_to_csv(self, filename):
        """
        Guarda los datos de la marcha en un archivo CSV.
        :param filename: Nombre del archivo CSV.
        """
        if not self.data:
            print("No hay datos disponibles para guardar.")
            return

        keys = self.data[0].keys()
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame'] + [f'{key}_x' for key in keys] + [f'{key}_y' for key in keys] +
                            [f'vel_{key}' for key in keys] + [f'acc_{key}' for key in keys] + ['activity'])
            for i, (points, velocities, accelerations) in enumerate(zip(self.data, self.angular_velocities, self.angular_accelerations)):
                row = [i]
                for key in keys:
                    row.extend([points[key][0], points[key][1]])
                for key in keys:
                    row.append(velocities.get(key, ''))
                for key in keys:
                    row.append(accelerations.get(key, ''))
                row.append(self.activities[i] if i < len(self.activities) else '')
                writer.writerow(row)

    def load_data_from_csv(self, filename):
        """
        Carga los datos de la marcha desde un archivo CSV.
        :param filename: Nombre del archivo CSV.
        """
        self.data = []
        self.angular_velocities = []
        self.angular_accelerations = []
        self.activities = []
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                points = {}
                velocities = {}
                accelerations = {}
                for key in row.keys():
                    if key.endswith('_x'):
                        point_key = key[:-2]
                        points[point_key] = (float(row[key]), float(row[f'{point_key}_y']))
                    elif key.startswith('vel_'):
                        vel_key = key[4:]
                        velocities[vel_key] = float(row[key])
                    elif key.startswith('acc_'):
                        acc_key = key[4:]
                        accelerations[acc_key] = float(row[key])
                    elif key == 'activity':
                        self.activities.append(int(row[key]) if row[key] else 0)
                self.data.append(points)
                self.angular_velocities.append(velocities)
                self.angular_accelerations.append(accelerations)

    def release_capture(self):
        """
        Libera la captura de video.
        """
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    analysis = GaitAnalysis(source=0)  # Cambia '0' por el ID de tu cámara o ruta del video
    analysis.run_analysis()
