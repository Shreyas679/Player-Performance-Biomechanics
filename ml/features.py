# ml/features.py
import json, numpy as np, pandas as pd
from pathlib import Path

ANGLE_PAIRS = {
    "r_elbow": ("r_shoulder","r_elbow","r_wrist"),
    "l_elbow": ("l_shoulder","l_elbow","l_wrist"),
    "r_knee": ("r_hip","r_knee","r_ankle"),
    "l_knee": ("l_hip","l_knee","l_ankle"),
    "r_shoulder_flex": ("neck","r_shoulder","r_elbow"),
    "l_shoulder_flex": ("neck","l_shoulder","l_elbow"),
    "trunk": ("r_hip","neck","l_hip"),
}

def angle(a,b,c):
    ba = a-b; bc = c-b
    nba = ba/ (np.linalg.norm(ba)+1e-6)
    nbc = bc/ (np.linalg.norm(bc)+1e-6)
    dot = np.clip(np.dot(nba, nbc), -1, 1)
    return np.degrees(np.arccos(dot))

def velocity(series, fps):
    v = np.diff(series, prepend=series[0]) * fps
    return v

def summarize(x):
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p10": float(np.percentile(x,10)),
        "p90": float(np.percentile(x,90)),
        "max": float(np.max(x)),
        "min": float(np.min(x)),
        "rom": float(np.max(x)-np.min(x))
    }

def extract_features(pose_json):
    joints = pose_json["joints"]
    frames = pose_json["frames"]
    fps = pose_json.get("fps", 24)
    J = {k: [] for k in joints}
    for fr in frames:
        for k in joints:
            J[k].append(fr[k])
    for k in J:
        J[k] = np.array(J[k], dtype=float)

    feats = {}
    # Angles and angular velocities
    for name,(a,b,c) in ANGLE_PAIRS.items():
        A = J[a]; B = J[b]; C = J[c]
        ang = np.array([angle(A[i],B[i],C[i]) for i in range(len(frames))])
        vel = velocity(ang, fps)
        for k,v in summarize(ang).items():
            feats[f"{name}_angle_{k}"] = v
        for k,v in summarize(vel).items():
            feats[f"{name}_angvel_{k}"] = v

    # Symmetry indices (elbow, knee, shoulder)
    for pair in [("r_elbow","l_elbow"),("r_knee","l_knee"),("r_shoulder_flex","l_shoulder_flex")]:
        r = np.array([angle(J[ANGLE_PAIRS[pair[0]][0]][i],
                            J[ANGLE_PAIRS[pair[0]][1]][i],
                            J[ANGLE_PAIRS[pair[0]][2]][i]) for i in range(len(frames))])
        l = np.array([angle(J[ANGLE_PAIRS[pair[1]][0]][i],
                            J[ANGLE_PAIRS[pair[1]][1]][i],
                            J[ANGLE_PAIRS[pair[1]][2]][i]) for i in range(len(frames))])
        si = (np.abs(r-l))/(np.maximum(r,l)+1e-6)
        for k,v in summarize(si).items():
            feats[f"sym_{pair[0]}_{pair[1]}_{k}"] = v

    # Postural stability (neck horizontal sway)
    neck_x = J["neck"][:,0]
    sway = neck_x - np.mean(neck_x)
    for k,v in summarize(sway).items():
        feats[f"sway_neck_x_{k}"] = v

    # Temporal peaks
    # e.g., peak r_shoulder angular velocity timing
    rs = np.array([angle(J["neck"][i], J["r_shoulder"][i], J["r_elbow"][i]) for i in range(len(frames))])
    rs_vel = velocity(rs, fps)
    peak_t = int(np.argmax(np.abs(rs_vel)))/max(1,len(frames))
    feats["r_shoulder_angvel_peak_time"] = float(peak_t)

    return feats

def batch_extract(keypoints_dir, out_csv):
    rows = []
    for p in Path(keypoints_dir).glob("*.json"):
        with open(p,"r") as f:
            d = json.load(f)
        feats = extract_features(d)
        feats["clip_id"] = p.stem
        rows.append(feats)
    df = pd.DataFrame(rows).set_index("clip_id")
    df.to_csv(out_csv)
    print(f"Saved {out_csv}")
