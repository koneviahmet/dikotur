"""
Duruş Tespiti Componenti
MediaPipe kullanarak oturan kişinin duruşunu analiz eder
"""

import cv2
import mediapipe as mp
import numpy as np

class PostureDetector:
    def __init__(self, show_text=False):
        try:
            # MediaPipe pose modelini başlat - Çocuklar için optimize edilmiş
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # Lower complexity to avoid TensorFlow Lite issues
                enable_segmentation=False,
                min_detection_confidence=0.3,  # Daha düşük eşik (çocuklar için)
                min_tracking_confidence=0.3   # Daha düşük eşik (çocuklar için)
            )
            self.mp_drawing = mp.solutions.drawing_utils
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MediaPipe Pose: {e}")
        
        # Ekranda yazı gösterme değişkeni
        self.show_text = show_text
        
        # Duruş skoru için değişkenler
        self.posture_score = 0
        self.is_good_posture = True
        self.frame_count = 0
        
        # Çocuklar için daha esnek eşikler
        self.posture_threshold = 30  # Çocuklar için daha düşük eşik
        self.smoothing_frames = 5  # Daha fazla yumuşatma (çocuklar daha hareketli)
        self.recent_scores = []
        
        # Çocuk tespiti için değişkenler
        self.is_child_detected = False
        self.child_confidence = 0
        
    def analyze_posture(self, frame):
        """
        Frame'deki kişinin duruşunu analiz eder
        Returns: (frame_with_landmarks, posture_score, is_good_posture)
        """
        # RGB'ye çevir (MediaPipe RGB bekler)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pose detection
        results = self.pose.process(rgb_frame)
        
        # Frame'i tekrar BGR'ye çevir
        frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            # Landmark'ları çiz (sadece show_text True ise)
            if self.show_text:
                self.mp_drawing.draw_landmarks(
                    frame_bgr, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS
                )
            
            # Duruş analizi yap
            posture_data = self._calculate_posture_score(results.pose_landmarks)
            
            # Skor yumuşatma (son 5 frame'in ortalaması)
            self.recent_scores.append(posture_data['score'])
            if len(self.recent_scores) > self.smoothing_frames:
                self.recent_scores.pop(0)
            
            # Yumuşatılmış skor
            smoothed_score = sum(self.recent_scores) / len(self.recent_scores)
            smoothed_is_good = smoothed_score >= self.posture_threshold
            
            self.posture_score = int(smoothed_score)
            self.is_good_posture = smoothed_is_good
            
            # Güncellenmiş veri ile frame'e yaz
            posture_data['score'] = self.posture_score
            posture_data['is_good'] = self.is_good_posture
            self._draw_posture_info(frame_bgr, posture_data)
            
        else:
            # Kişi tespit edilmedi
            self.posture_score = 0
            self.is_good_posture = False
            if self.show_text:
                cv2.putText(frame_bgr, "No person detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame_bgr, self.posture_score, self.is_good_posture
    
    def _detect_child(self, landmarks):
        """
        Çocuk tespiti yapar (boyut ve oranlara göre)
        """
        # Omuz genişliği
        left_shoulder = landmarks.landmark[11]
        right_shoulder = landmarks.landmark[12]
        shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        
        # Kalça genişliği
        left_hip = landmarks.landmark[23]
        right_hip = landmarks.landmark[24]
        hip_width = abs(right_hip.x - left_hip.x)
        
        # Omuz-kalça mesafesi
        shoulder_hip_distance = abs(left_shoulder.y - left_hip.y)
        
        # Çocuk tespiti kriterleri
        is_child = (
            shoulder_width < 0.15 or  # Dar omuzlar
            hip_width < 0.12 or       # Dar kalçalar
            shoulder_hip_distance < 0.2  # Kısa gövde
        )
        
        return is_child

    def _calculate_posture_score(self, landmarks):
        """
        Landmark'lardan duruş skorunu hesaplar (çocuklar için optimize)
        """
        # Çocuk tespiti
        is_child = self._detect_child(landmarks)
        self.is_child_detected = is_child
        
        # Gerekli noktaları al
        left_shoulder = landmarks.landmark[11]   # Sol omuz
        right_shoulder = landmarks.landmark[12]  # Sağ omuz
        left_hip = landmarks.landmark[23]        # Sol kalça
        right_hip = landmarks.landmark[24]       # Sağ kalça
        nose = landmarks.landmark[0]             # Burun
        
        # 1. Omuz hizalaması (en önemli)
        shoulder_angle = abs(np.arctan2(
            right_shoulder.y - left_shoulder.y,
            right_shoulder.x - left_shoulder.x
        ) * 180 / np.pi)
        
        # 2. Sırt dikliği (omuz-kalça hizalaması)
        spine_angle = abs(np.arctan2(
            left_shoulder.y - left_hip.y,
            left_shoulder.x - left_hip.x
        ) * 180 / np.pi)
        
        # 3. Baş pozisyonu (burun-omuz hizalaması)
        head_angle = abs(np.arctan2(
            nose.y - left_shoulder.y,
            nose.x - left_shoulder.x
        ) * 180 / np.pi)
        
        # Çocuklar için daha esnek skor hesaplama
        if is_child:
            # Çocuklar için çok daha esnek eşikler
            shoulder_score = max(0, 100 - (shoulder_angle * 3.0))  # Çok daha esnek
            spine_score = max(0, 100 - abs(90 - spine_angle) * 3.0)  # Çok daha esnek
            head_score = max(0, 100 - abs(90 - head_angle) * 1.5)  # Çok daha esnek
            
            # Çocuklar için farklı ağırlık dağılımı
            total_score = (shoulder_score * 0.3 + spine_score * 0.5 + head_score * 0.2)
            
            # Çocuklar için çok düşük eşik
            child_threshold = 20
            is_good = total_score >= child_threshold
        else:
            # Yetişkinler için normal eşikler
            shoulder_score = max(0, 100 - (shoulder_angle * 5.0))
            spine_score = max(0, 100 - abs(90 - spine_angle) * 5.0)
            head_score = max(0, 100 - abs(90 - head_angle) * 2.0)
            
            total_score = (shoulder_score * 0.4 + spine_score * 0.4 + head_score * 0.2)
            is_good = total_score >= self.posture_threshold
        
        return {
            'score': int(total_score),
            'raw_score': int(total_score),  # Ham skor için
            'is_good': is_good,
            'is_child': is_child,
            'shoulder_angle': shoulder_angle,
            'spine_angle': spine_angle,
            'head_angle': head_angle,
            'shoulder_score': int(shoulder_score),
            'spine_score': int(spine_score),
            'head_score': int(head_score)
        }
    
    def _draw_posture_info(self, frame, posture_data):
        """
        Duruş bilgilerini frame'e çizer
        """
        # Eğer yazı gösterme kapalıysa hiçbir şey çizme
        if not self.show_text:
            return
        
        # Skor ve durum
        score = posture_data['score']
        is_good = posture_data['is_good']
        
        # Renk belirle
        color = (0, 255, 0) if is_good else (0, 0, 255)  # Yeşil/Kırmızı
        status_text = "GOOD POSTURE" if is_good else "POOR POSTURE"
        
        # Ana bilgileri yaz
        cv2.putText(frame, f"Score: {score}/100", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, status_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Çocuk tespiti bilgisi
        if posture_data.get('is_child', False):
            cv2.putText(frame, "CHILD DETECTED", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)  # Sarı renk
        else:
            cv2.putText(frame, "ADULT DETECTED", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # Beyaz renk
        
        # Debug bilgileri (geçici)
        threshold_text = "Child: 20" if posture_data.get('is_child', False) else f"Adult: {self.posture_threshold}"
        cv2.putText(frame, f"Threshold: {threshold_text}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Raw: {posture_data.get('raw_score', 'N/A')}", (10, 170), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Detaylı bilgiler (küçük font) - Çocuk tespiti sonrası konumlandır
        y_offset = 120 if posture_data.get('is_child', False) else 110
        cv2.putText(frame, f"Shoulder: {posture_data['shoulder_score']}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Spine: {posture_data['spine_score']}", (10, y_offset + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Head: {posture_data['head_score']}", (10, y_offset + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def set_show_text(self, show_text):
        """
        Ekranda yazı gösterme durumunu ayarlar
        """
        self.show_text = show_text
    
    def get_posture_status(self):
        """
        Mevcut duruş durumunu döndürür
        """
        try:
            return {
                'score': self.posture_score,
                'is_good': self.is_good_posture
            }
        except Exception as e:
            print(f"⚠️  Error getting posture status: {e}")
            return {
                'score': 0,
                'is_good': False
            }
    
    def cleanup(self):
        """
        Kaynakları temizle
        """
        if hasattr(self, 'pose') and self.pose is not None:
            try:
                self.pose.close()
            except Exception as e:
                print(f"⚠️  Error closing MediaPipe pose: {e}")
