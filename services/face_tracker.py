import cv2
import mediapipe as mp
import numpy as np


class FaceTracker:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        # Use model_selection=0 (short-range) for better performance
        # Increase min_detection_confidence to reduce false positives
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        # Cache for detected faces to avoid reprocessing
        self.face_cache = {}
        print("🎯 Initialized intelligent face tracking with MediaPipe (optimized)")

    def detect_faces_in_frame(self, frame, frame_time=None):
        # Use cache if available
        if frame_time is not None and frame_time in self.face_cache:
            return self.face_cache[frame_time]
            
        try:
            # Resize frame for faster processing (half size)
            h, w, _ = frame.shape
            scale = 0.5
            small_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            
            # Convert to RGB (required by MediaPipe)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)

            faces = []
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)  # Scale back to original size
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)

                    center_x = x + width // 2
                    center_y = y + height // 2
                    confidence = detection.score[0]

                    faces.append({
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': width,
                        'height': height,
                        'confidence': confidence,
                        'area': width * height
                    })

            result = sorted(faces, key=lambda f: f['confidence'] * f['area'], reverse=True)
            
            # Cache the result
            if frame_time is not None:
                self.face_cache[frame_time] = result
                
            return result
        except Exception as e:
            print(f"    ⚠️ Face detection error: {e}")
            return []

    def smooth_trajectory(self, positions, window_size=5):
        if len(positions) <= window_size:
            return positions

        smoothed = []
        for i in range(len(positions)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(positions), i + window_size // 2 + 1)
            window = positions[start_idx:end_idx]

            avg_x = sum(pos[0] for pos in window) / len(window)
            avg_y = sum(pos[1] for pos in window) / len(window)
            smoothed.append((avg_x, avg_y))

        return smoothed

    def track_and_crop(self, clip):
        width, height = clip.size
        target_width = int(height * 9 / 16)
        if target_width % 2 != 0:
            target_width -= 1
        if width <= target_width:
            print("    ⏩ Skipping face tracking - video already in target aspect ratio")
            return clip

        print("    🎯 Analyzing frames for optimal face tracking (optimized)...")

        # Clear cache for new clip
        self.face_cache = {}
        
        face_positions = []
        # Analyze fewer frames for better performance
        # For short clips (<10s), analyze 6 frames, for longer clips analyze up to 8 frames
        num_samples = min(6, max(3, int(clip.duration / 3)))
        if clip.duration > 10:
            num_samples = min(8, max(6, int(clip.duration / 4)))
        
        print(f"    ⏳ Analyzing {num_samples} frames across {clip.duration:.1f}s of video...")
            
        sample_times = np.linspace(0, clip.duration, num_samples)

        for i, t in enumerate(sample_times):
            try:
                print(f"    ⏳ Processing frame {i+1}/{num_samples} at {t:.2f}s...")
                frame = clip.get_frame(t)
                faces = self.detect_faces_in_frame(frame, frame_time=t)

                if faces:
                    best_face = faces[0]
                    face_positions.append(best_face['center_x'])
                    print(f"    ✅ Frame {i+1}: Found face at position {best_face['center_x']} with confidence {best_face['confidence']:.2f}")
                else:
                    if face_positions:
                        face_positions.append(face_positions[-1])
                    else:
                        face_positions.append(width // 2)
                    print(f"    ⚠️ Frame {i+1}: No faces detected, using fallback position")

            except Exception as e:
                print(f"    ⚠️ Error processing frame {i+1} at {t:.2f}s: {e}")
                if face_positions:
                    face_positions.append(face_positions[-1])
                else:
                    face_positions.append(width // 2)

        if face_positions:
            print("    ⏳ Calculating optimal tracking trajectory...")
            positions = [(pos, height // 2) for pos in face_positions]
            # Use a smaller window size for smoother tracking
            smoothed_positions = self.smooth_trajectory(positions, window_size=3)
            # Use median for more stable center position
            center_x = int(np.median([pos[0] for pos in smoothed_positions]))
            print(f"    ✅ Face tracking complete, optimal center: {center_x}")
        else:
            center_x = width // 2
            print("    ⚠️  No faces detected, using center crop")

        # Ensure the crop area stays within the video boundaries
        center_x = max(target_width // 2, min(width - target_width // 2, center_x))
        left = center_x - target_width // 2
        
        print(f"    ⏳ Cropping video to {target_width}x{height} (9:16 ratio) at x-position: {left}")
        
        # Clear cache to free memory
        self.face_cache = {}
        
        # Use faster cropping with resize_algorithm='fast_bilinear' for better performance
        cropped_clip = clip.crop(x1=left, width=target_width, resize_algorithm='fast_bilinear')
        print(f"    ✅ Video cropping complete: {target_width}x{height}")
        return cropped_clip

    def close(self):
        """Release resources used by the face detector"""
        try:
            # Clear cache to free memory
            self.face_cache = {}
            # Close the face detection model
            self.face_detection.close()
            print("🎯 Face tracking resources released")
        except Exception as e:
            print(f"⚠️ Error closing face tracker: {e}")
