"""
Duruş Tespiti Componenti
MediaPipe kullanarak oturan kişinin duruşunu analiz eder
"""

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe pose küçük karelerde (özellikle Windows önizleme boyutunda) zayıflar
POSE_MIN_WIDTH = 640


class PostureDetector:
    def __init__(self, show_text=False):
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.mp_drawing = mp.solutions.drawing_utils
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MediaPipe Pose: {e}")

        self.show_text = show_text
        self.posture_score = 0
        self.is_good_posture = True
        self.frame_count = 0

        # Önden webcam / masa başı: öne eğilme daha sıkı yakalanır
        self.posture_threshold = 52
        self.smoothing_frames = 5
        self.recent_scores = []
        self.recent_good = []

        self.is_child_detected = False

    def _prepare_pose_frame(self, frame):
        """Pose için yeterli çözünürlük (landmark kalitesi)."""
        h, w = frame.shape[:2]
        if w >= POSE_MIN_WIDTH:
            return frame
        scale = POSE_MIN_WIDTH / w
        new_w = POSE_MIN_WIDTH
        new_h = max(1, int(h * scale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _lm_visible(lm, min_vis=0.5):
        return getattr(lm, "visibility", 1.0) >= min_vis

    def analyze_posture(self, frame):
        """
        Frame'deki kişinin duruşunu analiz eder.
        Returns: (frame_with_landmarks, posture_score, is_good_posture)
        """
        pose_frame = self._prepare_pose_frame(frame)
        rgb_pose = cv2.cvtColor(pose_frame, cv2.COLOR_BGR2RGB)
        rgb_pose = np.ascontiguousarray(rgb_pose)

        results = self.pose.process(rgb_pose)

        frame_bgr = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if results.pose_landmarks:
            if self.show_text:
                self.mp_drawing.draw_landmarks(
                    frame_bgr,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                )

            posture_data = self._calculate_posture_score(results.pose_landmarks)

            self.recent_scores.append(posture_data["score"])
            self.recent_good.append(1 if posture_data["is_good"] else 0)
            if len(self.recent_scores) > self.smoothing_frames:
                self.recent_scores.pop(0)
                self.recent_good.pop(0)

            smoothed_score = sum(self.recent_scores) / len(self.recent_scores)
            # Çoğunluk oylaması: tek karelik gürültüyü azaltır, eğik duruşu yutmasın
            good_votes = sum(self.recent_good)
            smoothed_is_good = (
                smoothed_score >= self.posture_threshold
                and good_votes >= (len(self.recent_good) + 1) // 2
            )

            self.posture_score = int(smoothed_score)
            self.is_good_posture = smoothed_is_good

            posture_data["score"] = self.posture_score
            posture_data["is_good"] = self.is_good_posture
            self._draw_posture_info(frame_bgr, posture_data)
        else:
            self.posture_score = 0
            self.is_good_posture = False
            if self.show_text:
                cv2.putText(
                    frame_bgr,
                    "No person detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        return frame_bgr, self.posture_score, self.is_good_posture

    def _detect_child(self, landmarks):
        """
        Yalnızca kalça landmark'ları güvenilirse çocuk modu (küçük pencerede yanlış pozitif önlenir).
        """
        lh = landmarks.landmark[23]
        rh = landmarks.landmark[24]
        if not (self._lm_visible(lh) and self._lm_visible(rh)):
            return False

        left_shoulder = landmarks.landmark[11]
        right_shoulder = landmarks.landmark[12]
        shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        hip_width = abs(rh.x - lh.x)
        shoulder_hip_distance = abs(
            (left_shoulder.y + right_shoulder.y) / 2 - (lh.y + rh.y) / 2
        )

        return (
            shoulder_width < 0.12
            and hip_width < 0.10
            and shoulder_hip_distance < 0.18
        )

    def _calculate_posture_score(self, landmarks):
        """
        Önden webcam için: öne eğilme, baş öne, omuz hizası.
        Yan açı metrikleri tek başına öne eğilmeyi yakalamaz.
        """
        is_child = self._detect_child(landmarks)
        self.is_child_detected = is_child

        nose = landmarks.landmark[0]
        left_shoulder = landmarks.landmark[11]
        right_shoulder = landmarks.landmark[12]
        left_hip = landmarks.landmark[23]
        right_hip = landmarks.landmark[24]

        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        shoulder_width = max(abs(right_shoulder.x - left_shoulder.x), 0.05)

        # Baş omuz hizasına ne kadar yaklaştı (öne eğilme / çökme)
        head_drop = nose.y - shoulder_mid_y
        head_drop_norm = head_drop / shoulder_width

        # Boyun öne eğikliği (dik: burun omuzların üstünde, dy < 0)
        neck_dx = nose.x - shoulder_mid_x
        neck_dy = nose.y - shoulder_mid_y
        if neck_dy < -0.02:
            neck_tilt = abs(np.degrees(np.arctan2(abs(neck_dx), abs(neck_dy))))
        else:
            neck_tilt = 90.0

        shoulder_angle = abs(
            np.degrees(
                np.arctan2(
                    right_shoulder.y - left_shoulder.y,
                    right_shoulder.x - left_shoulder.x,
                )
            )
        )

        spine_tilt = 0.0
        hips_visible = self._lm_visible(left_hip) and self._lm_visible(right_hip)
        if hips_visible:
            hip_mid_y = (left_hip.y + right_hip.y) / 2
            hip_mid_x = (left_hip.x + right_hip.x) / 2
            spine_tilt = abs(
                np.degrees(
                    np.arctan2(
                        shoulder_mid_y - hip_mid_y,
                        shoulder_mid_x - hip_mid_x,
                    )
                )
            )

        # Ceza puanları (yüksek = kötü duruş)
        forward_penalty = min(100.0, max(0.0, (head_drop_norm - 0.12) * 95.0))
        neck_penalty = min(100.0, max(0.0, (neck_tilt - 16.0) * 2.8))
        shoulder_penalty = min(100.0, shoulder_angle * 4.5)
        spine_penalty = (
            min(100.0, max(0.0, abs(90.0 - spine_tilt) * 4.0)) if hips_visible else 0.0
        )

        if is_child:
            total_penalty = (
                forward_penalty * 0.4
                + neck_penalty * 0.35
                + shoulder_penalty * 0.15
                + spine_penalty * 0.1
            )
            threshold = 38
        else:
            total_penalty = (
                forward_penalty * 0.42
                + neck_penalty * 0.38
                + shoulder_penalty * 0.12
                + spine_penalty * 0.08
            )
            threshold = self.posture_threshold

        total_score = max(0, min(100, int(100 - total_penalty)))
        is_good = total_score >= threshold

        return {
            "score": total_score,
            "raw_score": total_score,
            "is_good": is_good,
            "is_child": is_child,
            "head_drop_norm": round(head_drop_norm, 3),
            "neck_tilt": round(neck_tilt, 1),
            "shoulder_angle": round(shoulder_angle, 1),
            "spine_tilt": round(spine_tilt, 1),
            "forward_penalty": int(forward_penalty),
            "neck_penalty": int(neck_penalty),
            "shoulder_penalty": int(shoulder_penalty),
        }

    def _draw_posture_info(self, frame, posture_data):
        if not self.show_text:
            return

        score = posture_data["score"]
        is_good = posture_data["is_good"]
        color = (0, 255, 0) if is_good else (0, 0, 255)
        status_text = "GOOD POSTURE" if is_good else "POOR POSTURE"

        cv2.putText(
            frame, f"Score: {score}/100", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )
        cv2.putText(
            frame, status_text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )
        cv2.putText(
            frame,
            f"head:{posture_data.get('head_drop_norm')} neck:{posture_data.get('neck_tilt')}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )

    def set_show_text(self, show_text):
        self.show_text = show_text

    def get_posture_status(self):
        try:
            return {
                "score": self.posture_score,
                "is_good": self.is_good_posture,
            }
        except Exception as e:
            print(f"⚠️  Error getting posture status: {e}")
            return {"score": 0, "is_good": False}

    def cleanup(self):
        if hasattr(self, "pose") and self.pose is not None:
            try:
                self.pose.close()
            except Exception as e:
                print(f"⚠️  Error closing MediaPipe pose: {e}")
