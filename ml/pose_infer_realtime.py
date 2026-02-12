# ml/pose_infer_realtime.py
import cv2, numpy as np
import mediapipe as mp

from .pose_infer import PROJECT_JOINTS, MP

class RealtimePose:
    def __init__(self, image_width, image_height, alpha=0.3):
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
        self.w, self.h = image_width, image_height
        self.alpha = alpha
        self.ema = {k: None for k in PROJECT_JOINTS}

    def _xy(self, lm, name):
        p = lm[MP[name]]
        return np.array([float(p.x*self.w), float(p.y*self.h)], dtype=float)

    def detect_one(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks.landmark
        l_sh = self._xy(lm, "l_shoulder"); r_sh = self._xy(lm, "r_shoulder")
        neck = (l_sh + r_sh)/2.0
        sample = {
            "nose": self._xy(lm, "nose"),
            "neck": neck,
            "r_shoulder": r_sh, "r_elbow": self._xy(lm, "r_elbow"), "r_wrist": self._xy(lm, "r_wrist"),
            "l_shoulder": l_sh, "l_elbow": self._xy(lm, "l_elbow"), "l_wrist": self._xy(lm, "l_wrist"),
            "r_hip": self._xy(lm, "r_hip"), "r_knee": self._xy(lm, "r_knee"), "r_ankle": self._xy(lm, "r_ankle"),
            "l_hip": self._xy(lm, "l_hip"), "l_knee": self._xy(lm, "l_knee"), "l_ankle": self._xy(lm, "l_ankle"),
        }
        # EMA smoothing and scale normalization
        shw = np.linalg.norm(r_sh - l_sh) or 1.0
        for k, v in sample.items():
            v_norm = v / shw
            if self.ema[k] is None:
                self.ema[k] = v_norm
            else:
                self.ema[k] = self.alpha*v_norm + (1-self.alpha)*self.ema[k]
            sample[k] = self.ema[k].tolist()
        return sample
