# ml/pose_infer.py (real implementation with MediaPipe as example)
import cv2, json, numpy as np
from pathlib import Path

try:
    import mediapipe as mp
except ImportError:
    mp = None

PROJECT_JOINTS = ["nose","neck","r_shoulder","r_elbow","r_wrist","l_shoulder","l_elbow","l_wrist",
                  "r_hip","r_knee","r_ankle","l_hip","l_knee","l_ankle"]

# MediaPipe Pose indices: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
MP = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28,
}

def smooth_series(arr, win=5):
    if len(arr) < 3 or win < 3: return arr
    win = min(win, len(arr) - (1 - len(arr)%2))  # ensure odd, <= len
    if win % 2 == 0: win += 1
    # simple moving average
    kernel = np.ones(win)/win
    xs = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), 0, arr)
    return xs

def infer_pose_from_video(video_path, out_json_path=None, target_fps=24):
    if mp is None:
        raise RuntimeError("Install mediapipe: pip install mediapipe")
    cap = cv2.VideoCapture(str(video_path))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    step = max(1, int(round(orig_fps/target_fps)))
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)

    frames = []
    idx = 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % step != 0:
            idx += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            def xy(name):
                p = lm[MP[name]]
                return [float(p.x*w), float(p.y*h)]
            l_sh = xy("l_shoulder"); r_sh = xy("r_shoulder")
            neck = [(l_sh[0]+r_sh[0])/2.0, (l_sh[1]+r_sh[1])/2.0]
            sample = {
                "nose": xy("nose"),
                "neck": neck,
                "r_shoulder": xy("r_shoulder"), "r_elbow": xy("r_elbow"), "r_wrist": xy("r_wrist"),
                "l_shoulder": xy("l_shoulder"), "l_elbow": xy("l_elbow"), "l_wrist": xy("l_wrist"),
                "r_hip": xy("r_hip"), "r_knee": xy("r_knee"), "r_ankle": xy("r_ankle"),
                "l_hip": xy("l_hip"), "l_knee": xy("l_knee"), "l_ankle": xy("l_ankle"),
            }
        else:
            # if no detection, repeat last or skip; here we skip to avoid corrupt stats
            idx += 1
            continue
        frames.append(sample)
        idx += 1
    cap.release()

    # Optional: smooth and scale-normalize
    if len(frames)==0:
        raise RuntimeError("No pose detected in video")
    joints = PROJECT_JOINTS
    M = {k: np.array([f[k] for f in frames], dtype=float) for k in joints}

    # scale normalization by average shoulder width
    shw = np.linalg.norm(M["r_shoulder"] - M["l_shoulder"], axis=1)
    scale = np.median(shw[shw>0]) if np.any(shw>0) else 1.0
    for k in joints:
        M[k] = smooth_series(M[k], win=5) / max(scale, 1e-6)

    frames_out = [{k: [float(M[k][i,0]), float(M[k][i,1])] for k in joints} for i in range(len(frames))]
    out = {"joints": joints, "frames": frames_out, "fps": float(target_fps)}
    if out_json_path:
        Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w") as f:
            json.dump(out, f)
    return out

def load_pose_json(p):
    import json
    with open(p,"r") as f:
        return json.load(f)