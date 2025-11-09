import cv2
import numpy as np
import time
from collections import deque

class PowerfulEmotionDetector:
    def __init__(self):
        # Load multiple cascade classifiers for better detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Emotion history for smoothing
        self.emotion_history = deque(maxlen=10)
        self.confidence_history = deque(maxlen=10)
        
        # Frame counter for FPS
        self.frame_count = 0
        self.fps = 0
        self.prev_time = time.time()
        
        print("🚀 POWERFUL Emotion Detector Loaded (No DLIB Required)!")
    
    def calculate_facial_features(self, gray_face, color_face):
        """Calculate advanced facial features using OpenCV only"""
        h, w = gray_face.shape
        features = {
            'smile_intensity': 0,
            'eye_openness': 0,
            'brightness': 0,
            'contrast': 0,
            'face_symmetry': 0,
            'mouth_corners': 0
        }
        
        # Calculate brightness and contrast
        features['brightness'] = np.mean(gray_face)
        features['contrast'] = np.std(gray_face)
        
        # Detect smiles with multiple parameters
        smile_params = [
            (1.7, 20),  # Less sensitive
            (1.8, 25),  # Medium sensitivity  
            (1.9, 30)   # More sensitive
        ]
        
        total_smiles = 0
        for scale, neighbors in smile_params:
            smiles = self.smile_cascade.detectMultiScale(
                gray_face, 
                scaleFactor=scale, 
                minNeighbors=neighbors,
                minSize=(20, 20)
            )
            total_smiles += len(smiles)
        
        features['smile_intensity'] = min(total_smiles / len(smile_params), 3.0)
        
        # Detect eyes with multiple parameters
        eye_params = [
            (1.1, 3),   # Less sensitive
            (1.2, 4),   # Medium sensitivity
            (1.3, 5)    # More sensitive
        ]
        
        total_eyes = 0
        for scale, neighbors in eye_params:
            eyes = self.eye_cascade.detectMultiScale(
                gray_face,
                scaleFactor=scale,
                minNeighbors=neighbors,
                minSize=(15, 15)
            )
            total_eyes += len(eyes)
        
        features['eye_openness'] = min(total_eyes / len(eye_params), 4.0)
        
        # Calculate face symmetry (simplified)
        if w > 0:
            left_half = gray_face[:, :w//2]
            right_half = gray_face[:, w//2:]
            right_half_flipped = cv2.flip(right_half, 1)
            
            # Resize to same dimensions for comparison
            min_height = min(left_half.shape[0], right_half_flipped.shape[0])
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            
            if min_height > 0 and min_width > 0:
                left_resized = cv2.resize(left_half[:min_height, :min_width], (50, 50))
                right_resized = cv2.resize(right_half_flipped[:min_height, :min_width], (50, 50))
                
                # Calculate similarity
                similarity = np.corrcoef(left_resized.flatten(), right_resized.flatten())[0,1]
                features['face_symmetry'] = max(0, similarity) if not np.isnan(similarity) else 0.5
        
        return features
    
    def detect_emotion_advanced(self, gray_face, color_face):
        """Advanced emotion detection using multiple feature analysis"""
        features = self.calculate_facial_features(gray_face, color_face)
        
        # Extract features
        smile_intensity = features['smile_intensity']
        eye_openness = features['eye_openness']
        brightness = features['brightness']
        contrast = features['contrast']
        symmetry = features['face_symmetry']
        
        # Emotion scoring system
        emotion_scores = {
            'HAPPY': 0,
            'SAD': 0, 
            'ANGRY': 0,
            'SURPRISED': 0,
            'TIRED': 0,
            'NEUTRAL': 0
        }
        
        # Happy: Strong smile + normal eyes + good symmetry
        if smile_intensity > 1.5:
            emotion_scores['HAPPY'] += smile_intensity * 3
        if smile_intensity > 0.5:
            emotion_scores['HAPPY'] += smile_intensity * 2
        
        # Sad: Low smile + normal eyes + possible asymmetry
        if smile_intensity < 0.5 and eye_openness >= 1:
            emotion_scores['SAD'] += (1 - smile_intensity) * 2
        if symmetry < 0.3:
            emotion_scores['SAD'] += 1
        
        # Angry: Low smile + intense eyes + high contrast
        if smile_intensity < 0.3 and contrast > 40:
            emotion_scores['ANGRY'] += (40 / contrast) * 2
        if eye_openness >= 2 and smile_intensity < 0.5:
            emotion_scores['ANGRY'] += 1
        
        # Surprised: Many eyes detected + high brightness
        if eye_openness >= 2.5:
            emotion_scores['SURPRISED'] += eye_openness * 2
        if brightness > 100:
            emotion_scores['SURPRISED'] += 1
        
        # Tired: Few eyes + low brightness
        if eye_openness < 1:
            emotion_scores['TIRED'] += (2 - eye_openness) * 2
        if brightness < 50:
            emotion_scores['TIRED'] += 1
        
        # Neutral: Balanced features
        if 0.5 <= smile_intensity <= 1.5 and 1 <= eye_openness <= 2:
            emotion_scores['NEUTRAL'] += 2
        
        # Get emotion with highest score
        best_emotion = max(emotion_scores, key=emotion_scores.get)
        best_score = emotion_scores[best_emotion]
        
        # Calculate confidence (normalize to 0-1)
        total_score = sum(emotion_scores.values())
        confidence = min(best_score / max(total_score, 1), 0.95)
        
        # Apply minimum confidence threshold
        if confidence < 0.3:
            best_emotion = "NEUTRAL"
            confidence = 0.5
        
        # Apply temporal smoothing
        self.emotion_history.append(best_emotion)
        self.confidence_history.append(confidence)
        
        # Use mode of recent emotions for stability
        if len(self.emotion_history) >= 5:
            emotion_counts = {}
            for emotion in self.emotion_history:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            smoothed_emotion = max(emotion_counts, key=emotion_counts.get)
            if emotion_counts[smoothed_emotion] >= len(self.emotion_history) // 2:
                best_emotion = smoothed_emotion
                confidence = min(np.mean(list(self.confidence_history)) + 0.1, 0.9)
        
        return best_emotion, confidence, features
    
    def get_emotion_color(self, emotion):
        """Get vibrant colors for each emotion"""
        colors = {
            'HAPPY': (0, 255, 0),        # Bright Green
            'SAD': (255, 0, 0),          # Blue
            'ANGRY': (0, 0, 255),        # Red
            'SURPRISED': (0, 255, 255),  # Yellow
            'TIRED': (128, 0, 128),      # Purple
            'NEUTRAL': (255, 255, 255),  # White
            'EXCITED': (255, 165, 0)     # Orange
        }
        return colors.get(emotion, (255, 255, 255))
    
    def draw_advanced_dashboard(self, frame, x, y, w, h, emotion, confidence, features):
        """Draw advanced emotion dashboard with metrics"""
        color = self.get_emotion_color(emotion)
        
        # Draw enhanced face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Draw emotion badge with background
        text = f"{emotion}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        
        # Badge background
        badge_y = y - text_size[1] - 15
        cv2.rectangle(frame, (x, badge_y), (x + text_size[0] + 20, y), color, -1)
        cv2.rectangle(frame, (x, badge_y), (x + text_size[0] + 20, y), (255, 255, 255), 1)
        
        # Emotion text
        cv2.putText(frame, text, (x + 10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        # Confidence bar with percentage
        bar_width = 150
        bar_height = 12
        bar_x = x
        bar_y = y + h + 10
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        # Filled bar
        fill_width = int(confidence * bar_width)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 1)
        
        # Percentage text
        percent_text = f"Confidence: {confidence*100:.1f}%"
        cv2.putText(frame, percent_text, (bar_x, bar_y + bar_height + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Feature metrics (right side)
        metrics_x = x + w + 10
        metrics_y = y
        
        metrics = [
            f"Smile: {features['smile_intensity']:.1f}",
            f"Eyes: {features['eye_openness']:.1f}",
            f"Bright: {features['brightness']:.0f}",
            f"Contrast: {features['contrast']:.1f}",
            f"Symmetry: {features['face_symmetry']:.2f}"
        ]
        
        for i, metric in enumerate(metrics):
            cv2.putText(frame, metric, (metrics_x, metrics_y + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def update_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.prev_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.prev_time = current_time
    
    def run_detection(self):
        """Main emotion detection loop"""
        cap = cv2.VideoCapture(0)
        
        # Set camera resolution for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n🎭 POWERFUL EMOTION DETECTION SYSTEM")
        print("=======================================")
        print("😊 SMILE WIDELY → Happy")
        print("😮 OPEN EYES WIDE → Surprised") 
        print("😪 CLOSE EYES → Tired")
        print("😠 DON'T SMILE + INTENSE EYES → Angry")
        print("😢 SLIGHT SMILE → Sad")
        print("😐 NORMAL FACE → Neutral")
        print("⏹️  Press 'Q' to quit")
        print("=======================================\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Update FPS
            self.update_fps()
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Enhance image quality
            gray = cv2.equalizeHist(gray)
            
            # Detect faces with optimized parameters
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=6, 
                minSize=(80, 80),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            emotion_detected = False
            
            for (x, y, w, h) in faces:
                # Extract face region with padding
                padding = 15
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                
                face_roi_gray = gray[y1:y2, x1:x2]
                face_roi_color = frame[y1:y2, x1:x2]
                
                # Resize for consistent processing
                if face_roi_gray.shape[0] > 0 and face_roi_gray.shape[1] > 0:
                    face_roi_gray = cv2.resize(face_roi_gray, (200, 200))
                    
                    # Detect emotion with advanced analysis
                    emotion, confidence, features = self.detect_emotion_advanced(face_roi_gray, face_roi_color)
                    
                    # Draw advanced dashboard
                    self.draw_advanced_dashboard(frame, x, y, w, h, emotion, confidence, features)
                    emotion_detected = True
            
            # Display system information
            cv2.putText(frame, f"FPS: {self.fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display real-time instructions
            instructions = [
                "SMILE → HAPPY",
                "WIDE EYES → SURPRISED", 
                "CLOSE EYES → TIRED",
                "NO SMILE → ANGRY/SAD",
                "NORMAL → NEUTRAL"
            ]
            
            for i, instruction in enumerate(instructions):
                cv2.putText(frame, instruction, (10, 100 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display status message
            if not emotion_detected:
                cv2.putText(frame, "Position your face in the camera", 
                           (frame.shape[1]//2 - 150, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow('🎭 POWERFUL Emotion Detection - Press Q to quit', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n🎭 Emotion detection stopped.")

if __name__ == "__main__":
    detector = PowerfulEmotionDetector()
    detector.run_detection()