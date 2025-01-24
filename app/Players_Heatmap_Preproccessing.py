import sys
import traceback
import cv2
import numpy as np
import torch
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QListWidget, QListWidgetItem, QFileDialog, QMessageBox,QProgressDialog,QProgressBar)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.ndimage import gaussian_filter
import os
from sklearn.cluster import DBSCAN


class PlayerTracker:
    def __init__(self, max_players=26):
        self.max_players = max_players
        self.player_positions = {}  # Historical positions
        self.active_players = set()  # Currently active player IDs
        self.position_buffer = {}  # Recent positions for each track
        self.buffer_size = 30  # Number of frames to buffer
        self.min_track_frames = 10  # Minimum frames to consider track valid

    def update_tracks(self, detections, frame_dims):
        current_tracks = {}
        height, width = frame_dims

        # Update position buffers
        for detection in detections:
            track_id = int(detection[4])
            x1, y1, x2, y2 = detection[:4].tolist()

            # Filter out detections at image edges
            if (x1 <= 5 or x2 >= width - 5 or
                    y1 <= 5 or y2 >= height - 5):
                continue

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if track_id not in self.position_buffer:
                self.position_buffer[track_id] = []

            self.position_buffer[track_id].append((center_x, center_y))
            if len(self.position_buffer[track_id]) > self.buffer_size:
                self.position_buffer[track_id].pop(0)

            current_tracks[track_id] = (center_x, center_y)

        # Clean up old tracks
        for track_id in list(self.position_buffer.keys()):
            if track_id not in current_tracks:
                if len(self.position_buffer[track_id]) < self.min_track_frames:
                    del self.position_buffer[track_id]
                    if track_id in self.player_positions:
                        del self.player_positions[track_id]

        # Merge similar tracks using DBSCAN clustering
        if len(self.position_buffer) > self.max_players:
            self.merge_similar_tracks()

        # Update active players
        self.active_players = set(self.position_buffer.keys())

        return current_tracks

    def merge_similar_tracks(self):
        # Collect average positions for each track
        track_positions = []
        track_ids = []

        for track_id, positions in self.position_buffer.items():
            if len(positions) >= self.min_track_frames:
                avg_pos = np.mean(positions, axis=0)
                track_positions.append(avg_pos)
                track_ids.append(track_id)

        if not track_positions:
            return

        # Cluster tracks based on average positions
        clustering = DBSCAN(eps=50, min_samples=1).fit(track_positions)

        # Group tracks by cluster
        clusters = {}
        for track_id, cluster_id in zip(track_ids, clustering.labels_):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(track_id)

        # Merge tracks within each cluster
        for cluster_tracks in clusters.values():
            if len(cluster_tracks) > 1:
                main_track = min(cluster_tracks)
                for other_track in cluster_tracks[1:]:
                    # Merge positions into main track
                    if other_track in self.player_positions:
                        if main_track not in self.player_positions:
                            self.player_positions[main_track] = []
                        self.player_positions[main_track].extend(
                            self.player_positions[other_track]
                        )
                        del self.player_positions[other_track]

                    # Clean up buffers
                    if other_track in self.position_buffer:
                        del self.position_buffer[other_track]


class InitialProcessingThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.model = YOLO("yolov8m.pt")
        self.pitch_mapper = PitchMapper()
        self.player_tracker = PlayerTracker(max_players=26)
        self.heatmaps = {}

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception("Could not open video file")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.pitch_mapper.initialize_dimensions(frame_width, frame_height)

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model.track(frame, persist=True, classes=[0])
                if hasattr(results[0].boxes, 'data'):
                    detections = results[0].boxes.data
                    current_tracks = self.player_tracker.update_tracks(
                        detections, frame.shape[:2]
                    )

                    # Update pitch positions
                    for track_id, (center_x, center_y) in current_tracks.items():
                        pitch_x, pitch_y = self.pitch_mapper.map_to_pitch_coordinates(
                            center_x, center_y
                        )

                        player_id = f"Player {track_id}"
                        if player_id not in self.player_tracker.player_positions:
                            self.player_tracker.player_positions[player_id] = []

                        self.player_tracker.player_positions[player_id].append(
                            (pitch_x, pitch_y)
                        )

                frame_count += 1
                progress = int((frame_count / total_frames) * 100)
                self.progress_signal.emit(progress)

            # Generate and save heatmaps
            os.makedirs('heatmaps', exist_ok=True)

            # Only process tracks with sufficient data
            valid_players = {
                player_id: positions
                for player_id, positions in self.player_tracker.player_positions.items()
                if len(positions) >= self.player_tracker.min_track_frames
            }

            for player_id, positions in valid_players.items():
                heatmap = self.generate_heatmap(positions)
                self.save_heatmap(heatmap, player_id)
                self.heatmaps[player_id] = heatmap

            cap.release()
            self.finished_signal.emit(self.heatmaps)

        except Exception as e:
            self.error_signal.emit(str(e))

    def generate_heatmap(self, positions):
        heatmap = np.zeros((105, 68))
        positions = np.array(positions)
        x_indices = np.clip(positions[:, 0].astype(int), 0, 67)
        y_indices = np.clip(positions[:, 1].astype(int), 0, 104)

        for x, y in zip(x_indices, y_indices):
            heatmap[y, x] += 1

        heatmap = gaussian_filter(heatmap, sigma=2)
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)

        return heatmap

    def save_heatmap(self, heatmap, player_id):
        plt.figure(figsize=(10, 15))

        # Draw pitch lines first
        plt.plot([0, 68], [0, 0], 'white', linewidth=1)
        plt.plot([0, 0], [0, 105], 'white', linewidth=1)
        plt.plot([68, 68], [0, 105], 'white', linewidth=1)
        plt.plot([0, 68], [105, 105], 'white', linewidth=1)
        plt.plot([0, 68], [52.5, 52.5], 'white', linewidth=1)

        # Draw penalty areas
        plt.plot([13.85, 13.85], [0, 16.5], 'white', linewidth=1)
        plt.plot([54.15, 54.15], [0, 16.5], 'white', linewidth=1)
        plt.plot([13.85, 54.15], [16.5, 16.5], 'white', linewidth=1)
        plt.plot([13.85, 13.85], [88.5, 105], 'white', linewidth=1)
        plt.plot([54.15, 54.15], [88.5, 105], 'white', linewidth=1)
        plt.plot([13.85, 54.15], [88.5, 88.5], 'white', linewidth=1)

        # Draw goal areas
        plt.plot([24.85, 24.85], [0, 5.5], 'white', linewidth=1)
        plt.plot([43.15, 43.15], [0, 5.5], 'white', linewidth=1)
        plt.plot([24.85, 43.15], [5.5, 5.5], 'white', linewidth=1)
        plt.plot([24.85, 24.85], [99.5, 105], 'white', linewidth=1)
        plt.plot([43.15, 43.15], [99.5, 105], 'white', linewidth=1)
        plt.plot([24.85, 43.15], [99.5, 99.5], 'white', linewidth=1)

        # Draw center circle
        circle = plt.Circle((34, 52.5), 9.15, fill=False, color='white', linewidth=1)
        plt.gca().add_artist(circle)

        # Plot heatmap
        plt.imshow(heatmap, extent=[0, 68, 0, 105], origin='lower',
                   cmap='RdYlBu_r', alpha=0.6)

        plt.title(f"{player_id} Heatmap")
        plt.gca().set_aspect('equal')
        plt.savefig(f'heatmaps/{player_id.replace(" ", "_")}_heatmap.png',
                    bbox_inches='tight', facecolor='black')
        plt.close()


class PitchMapper:
    def __init__(self):
        # Standard football pitch dimensions in meters
        self.pitch_length = 105  # Length in meters
        self.pitch_width = 68  # Width in meters
        self.video_width = None
        self.video_height = None

    def initialize_dimensions(self, video_width, video_height):
        """Initialize video dimensions for coordinate mapping"""
        self.video_width = video_width
        self.video_height = video_height

    def map_to_pitch_coordinates(self, x, y):
        """
        Map video coordinates to pitch coordinates using simple linear mapping.
        This assumes the video is filmed from a side-on perspective.
        """
        if not self.video_width or not self.video_height:
            return 0, 0

        # Linear mapping from video coordinates to pitch coordinates
        # Adjust x mapping to account for perspective (closer to camera = wider)
        y_ratio = y / self.video_height
        perspective_factor = 1 + (0.5 - y_ratio) * 0.5  # Adjust perspective based on y position

        # Apply perspective correction to x coordinate
        adjusted_x = x / perspective_factor

        # Map to pitch coordinates
        pitch_x = (adjusted_x / self.video_width) * self.pitch_width
        pitch_y = (y / self.video_height) * self.pitch_length

        # Ensure coordinates are within pitch boundaries
        pitch_x = max(0, min(pitch_x, self.pitch_width))
        pitch_y = max(0, min(pitch_y, self.pitch_length))

        return pitch_x, pitch_y


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, list, dict)
    update_slider_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)
    update_heatmap_signal = pyqtSignal(str)

    def __init__(self, video_path, saved_heatmaps=None):
        super().__init__()
        self.video_path = video_path
        self.saved_heatmaps = saved_heatmaps or {}
        try:
            self.model = YOLO("yolov8m.pt")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.error_signal.emit("Error loading YOLO model. Please ensure yolov8m.pt is present.")
            return

        self.player_positions = {}
        self.current_frame_positions = {}
        self.player_ids = {}
        self.paused = False
        self.current_frame = 0
        self.selected_player = None
        self.original_frame = None
        self.pitch_mapper = PitchMapper()
        self.running = True

        # Initialize video properties
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video file")

            self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Initialize pitch mapper with video dimensions
            self.pitch_mapper.initialize_dimensions(self.video_width, self.video_height)

        except Exception as e:
            print(f"Error initializing video: {e}")
            self.error_signal.emit(f"Error loading video: {str(e)}")
            return

    def create_inpaint_mask(self, frame, detections, selected_id):
        """Create a mask for inpainting unselected players"""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for detection in detections:
            track_id = int(detection[4])
            player_id = f"Player {track_id}"

            if player_id != selected_id:
                x1, y1, x2, y2 = map(int, detection[:4])
                # Add padding to the mask for better inpainting
                padding = 5
                y1 = max(0, y1 - padding)
                y2 = min(frame.shape[0], y2 + padding)
                x1 = max(0, x1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                mask[y1:y2, x1:x2] = 255

        return mask

    def run(self):
        """Main video processing loop"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception("Could not open video file")

            while cap.isOpened() and self.running:
                if not self.paused:
                    ret, frame = cap.read()
                    if ret:
                        self.process_frame(frame)
                    else:
                        break
                else:
                    self.msleep(50)

            cap.release()

        except Exception as e:
            print(f"Error in video processing: {e}")
            self.error_signal.emit(f"Error processing video: {str(e)}")

    def process_frame(self, frame):
        """Process a single video frame"""
        try:
            self.original_frame = frame.copy()
            results = self.model.track(frame, persist=True, classes=[0])

            processed_frame = frame.copy()
            current_players = {}
            current_positions = {}

            if hasattr(results[0].boxes, 'data'):
                detections = results[0].boxes.data

                if self.selected_player:
                    mask = self.create_inpaint_mask(frame, detections, self.selected_player)
                    processed_frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

                for detection in detections:
                    self.process_detection(detection, processed_frame, current_players, current_positions)

            display_frame = processed_frame if self.selected_player else results[0].plot()
            self.current_frame_positions = current_positions
            self.change_pixmap_signal.emit(display_frame, list(current_players.keys()), current_positions)
            self.current_frame += 1
            self.update_slider_signal.emit(self.current_frame)

        except Exception as e:
            print(f"Error processing frame: {e}")
            self.error_signal.emit(f"Error processing frame: {str(e)}")

    def process_detection(self, detection, processed_frame, current_players, current_positions):
        """Process a single detection"""
        try:
            track_id = int(detection[4])
            x1, y1, x2, y2 = detection[:4].tolist()
            player_id = f"Player {track_id}"

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if player_id not in self.player_positions:
                self.player_positions[player_id] = []
                self.player_ids[track_id] = player_id

            pitch_x, pitch_y = self.pitch_mapper.map_to_pitch_coordinates(center_x, center_y)

            self.player_positions[player_id].append((pitch_x, pitch_y))
            current_players[player_id] = (pitch_x, pitch_y)
            current_positions[player_id] = (pitch_x, pitch_y)

            if player_id == self.selected_player:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_frame, f"{player_id} ({pitch_x:.1f}, {pitch_y:.1f})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                self.update_heatmap_signal.emit(player_id)

        except Exception as e:
            print(f"Error processing detection: {e}")

    def set_frame(self, frame_number):
        """Set video to specific frame"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                self.process_frame(frame)
            cap.release()
        except Exception as e:
            print(f"Error setting frame: {e}")
            self.error_signal.emit(f"Error setting frame: {str(e)}")

    def set_selected_player(self, player_id):
        self.selected_player = player_id

    def pause(self):
        self.paused = not self.paused

    def stop(self):
        """Stop the video thread"""
        self.running = False
        self.wait()


class FootballTrackerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Football Player Tracker")
        self.setGeometry(100, 100, 1280, 720)
        self.thread = None
        self.setup_error_handling()
        self.position_history = {}  # Store position history for heatmap
        self.heatmap_data = np.zeros((105, 68))  # Initialize heatmap data array
        self.frame_count = 0  # Track number of frames for averaging
        self.saved_heatmaps = {}


        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel
        left_layout = QVBoxLayout()
        self.player_list_widget = QListWidget()
        self.player_list_widget.itemClicked.connect(self.player_selected)
        left_layout.addWidget(QLabel("Players:"))
        left_layout.addWidget(self.player_list_widget)

        self.clear_selection_button = QPushButton("Show All Players")
        self.clear_selection_button.clicked.connect(self.clear_player_selection)
        left_layout.addWidget(self.clear_selection_button)

        main_layout.addLayout(left_layout)

        # Center panel (video)
        center_layout = QVBoxLayout()
        self.import_button = QPushButton("Import Video")
        self.import_button.clicked.connect(self.import_video)
        center_layout.addWidget(self.import_button)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(self.video_label)

        controls_layout = QHBoxLayout()
        self.play_pause_button = QPushButton("Pause")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        controls_layout.addWidget(self.play_pause_button)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self.set_frame)
        controls_layout.addWidget(self.slider)

        center_layout.addLayout(controls_layout)
        main_layout.addLayout(center_layout)

        # Right panel (heatmap)
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Heatmap:"))

        # Create matplotlib figure for heatmap
        plt.style.use('dark_background')
        self.figure = plt.figure(figsize=(6, 8), facecolor='#1e1e1e')
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        main_layout.addLayout(right_layout)

        # Set layout stretch factors
        main_layout.setStretch(0, 1)  # Player list
        main_layout.setStretch(1, 2)  # Video
        main_layout.setStretch(2, 1)  # Heatmap

        self.video_path = None
        self.thread = None
        self.current_player = None

        # Draw initial pitch
        self.draw_pitch()

    def setup_error_handling(self):
        """Setup error handling message box"""
        self.error_dialog = QMessageBox()
        self.error_dialog.setIcon(QMessageBox.Icon.Critical)
        self.error_dialog.setWindowTitle("Error")

    def show_error(self, message):
        """Display error message to user"""
        self.error_dialog.setText(message)
        self.error_dialog.exec()

    def draw_pitch(self):
        """Draw a vertical football pitch on the matplotlib axes"""
        self.ax.clear()

        # Set pitch dimensions (swapped for vertical orientation)
        pitch_width = 68
        pitch_length = 105

        # Set background color to green
        self.ax.set_facecolor('#2e8b57')

        # Draw pitch outline
        self.ax.plot([0, pitch_width], [0, 0], 'white', linewidth=1)
        self.ax.plot([0, 0], [0, pitch_length], 'white', linewidth=1)
        self.ax.plot([pitch_width, pitch_width], [0, pitch_length], 'white', linewidth=1)
        self.ax.plot([0, pitch_width], [pitch_length, pitch_length], 'white', linewidth=1)

        # Halfway line
        self.ax.plot([0, pitch_width], [pitch_length / 2, pitch_length / 2], 'white', linewidth=1)

        # Center circle
        center_circle = plt.Circle((pitch_width / 2, pitch_length / 2), 9.15, fill=False, color='white', linewidth=1)
        self.ax.add_artist(center_circle)

        # Center dot
        center_dot = plt.Circle((pitch_width / 2, pitch_length / 2), 0.5, color='white')
        self.ax.add_artist(center_dot)

        # Penalty areas
        self.ax.plot([13.85, 13.85], [0, 16.5], 'white', linewidth=1)
        self.ax.plot([54.15, 54.15], [0, 16.5], 'white', linewidth=1)
        self.ax.plot([13.85, 54.15], [16.5, 16.5], 'white', linewidth=1)

        self.ax.plot([13.85, 13.85], [pitch_length - 16.5, pitch_length], 'white', linewidth=1)
        self.ax.plot([54.15, 54.15], [pitch_length - 16.5, pitch_length], 'white', linewidth=1)
        self.ax.plot([13.85, 54.15], [pitch_length - 16.5, pitch_length - 16.5], 'white', linewidth=1)

        # Goal areas
        self.ax.plot([24.85, 24.85], [0, 5.5], 'white', linewidth=1)
        self.ax.plot([43.15, 43.15], [0, 5.5], 'white', linewidth=1)
        self.ax.plot([24.85, 43.15], [5.5, 5.5], 'white', linewidth=1)

        self.ax.plot([24.85, 24.85], [pitch_length - 5.5, pitch_length], 'white', linewidth=1)
        self.ax.plot([43.15, 43.15], [pitch_length - 5.5, pitch_length], 'white', linewidth=1)
        self.ax.plot([24.85, 43.15], [pitch_length - 5.5, pitch_length - 5.5], 'white', linewidth=1)

        # Set axis limits with minimal padding
        self.ax.set_xlim(-1, pitch_width + 1)
        self.ax.set_ylim(-1, pitch_length + 1)

        # Remove axis labels and ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Set aspect ratio to equal
        self.ax.set_aspect('equal')

        # Tight layout to remove extra padding
        self.figure.tight_layout()

    def import_video(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Import Video",
                "",
                "Video Files (*.mp4 *.avi *.mkv)"
            )

            if file_name:
                if self.thread is not None:
                    self.thread.stop()
                    self.thread.wait()

                self.video_path = file_name
                self.process_video_initially()

        except Exception as e:
            self.show_error(f"Error importing video: {str(e)}")

    def start_tracking(self):
        """Start video tracking thread with heatmap updates"""

        try:
            """Start video tracking with saved heatmaps"""
            if self.video_path:
                self.thread = VideoThread(self.video_path, self.saved_heatmaps)
                self.thread = VideoThread(self.video_path)
                self.thread.change_pixmap_signal.connect(self.update_image)
                self.thread.update_slider_signal.connect(self.update_slider)
                self.thread.error_signal.connect(self.show_error)
                self.thread.update_heatmap_signal.connect(self.update_heatmap)  # Connect new signal
                self.slider.setMaximum(self.thread.total_frames)
                self.thread.start()

        except Exception as e:
            self.show_error(f"Error starting tracking: {str(e)}")

    def process_video_initially(self):
        """Initial processing with progress bar"""
        self.progress_dialog = QProgressDialog("Processing video...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(True)

        self.init_thread = InitialProcessingThread(self.video_path)
        self.init_thread.progress_signal.connect(self.progress_dialog.setValue)
        self.init_thread.finished_signal.connect(self.initial_processing_finished)
        self.init_thread.error_signal.connect(self.show_error)
        self.init_thread.start()

    def initial_processing_finished(self, heatmaps):
        """Called when initial processing is complete"""
        self.saved_heatmaps = heatmaps
        self.start_tracking()

    def player_selected(self, item):
        """Handle player selection with heatmap reset"""
        player_id = item.text()
        if self.thread:
            if self.current_player != player_id:
                self.current_player = player_id
                self.thread.set_selected_player(player_id)
                # Clear existing heatmap data for the new player
                if player_id not in self.position_history:
                    self.position_history[player_id] = []
                self.reset_and_show_heatmap(player_id)

    def reset_and_show_heatmap(self, player_id):
        """Reset and initialize heatmap for new player selection"""
        self.draw_pitch()
        if player_id in self.position_history:
            self.update_heatmap(player_id)
        else:
            self.canvas.draw()

    def update_heatmap(self, player_id):
        """Update heatmap display using saved heatmap"""
        if player_id == self.current_player and player_id in self.saved_heatmaps:
            self.draw_pitch()

            # Display saved heatmap
            self.ax.imshow(
                self.saved_heatmaps[player_id],
                extent=[0, 68, 0, 105],
                origin='lower',
                cmap='RdYlBu_r',
                alpha=0.6,
                aspect='equal'
            )

            # Add current position if available
            if self.thread and player_id in self.thread.current_frame_positions:
                current_x, current_y = self.thread.current_frame_positions[player_id]
                self.ax.plot(current_x, current_y, 'wo', markersize=8, markeredgecolor='black')

            self.ax.set_title(f"{player_id} Movement Heatmap", color='white', pad=10)
            self.figure.tight_layout()
            self.canvas.draw()

    def clear_player_selection(self):
        """Clear player selection and heatmap"""
        if self.thread:
            self.current_player = None
            self.thread.set_selected_player(None)
            self.draw_pitch()
            self.canvas.draw()

    def update_image(self, cv_img, players, current_positions):
        """Update image and player list without redundant heatmap updates"""
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)
        self.update_player_list(players)

    def update_player_list(self, players):
        current_item = self.player_list_widget.currentItem()
        current_selected = current_item.text() if current_item else None

        self.player_list_widget.clear()
        for player in players:
            item = QListWidgetItem(player)
            self.player_list_widget.addItem(item)
            if player == current_selected:
                item.setSelected(True)
                self.player_list_widget.setCurrentItem(item)

    def toggle_play_pause(self):
        if self.thread:
            self.thread.pause()
            button_text = "Play" if self.thread.paused else "Pause"
            self.play_pause_button.setText(button_text)

    def set_frame(self, frame_number):
        if self.thread:
            self.thread.set_frame(frame_number)

    def update_slider(self, frame_number):
        self.slider.setValue(frame_number)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 360, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def closeEvent(self, event):
        """Handle application closing"""
        if self.thread is not None:
            self.thread.stop()
            self.thread.wait()
        event.accept()


def exception_hook(exctype, value, tb):
    print(''.join(traceback.format_exception(exctype, value, tb)))
    sys.exit(1)


sys.excepthook = exception_hook

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FootballTrackerGUI()
    window.show()
    sys.exit(app.exec())