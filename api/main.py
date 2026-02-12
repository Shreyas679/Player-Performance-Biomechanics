x# api/main.py
import json, base64, asyncio
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from xgboost import XGBRegressor, XGBClassifier

from ml.features import extract_features
from ml.pose_infer import infer_pose_from_video

# Optional realtime pose via MediaPipe
try:
    import mediapipe as mp
    _mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
    _mp_idx = {
        "nose": 0, "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
        "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24, "l_knee": 25, "r_knee": 26,
        "l_ankle": 27, "r_ankle": 28
    }
except Exception:
    _mp_pose = None
    _mp_idx = None

# Joints order expected by features
PROJECT_JOINTS = [
    "nose","neck","r_shoulder","r_elbow","r_wrist","l_shoulder","l_elbow","l_wrist",
    "r_hip","r_knee","r_ankle","l_hip","l_knee","l_ankle"
]

def detect_one_frame(frame):
    if _mp_pose is None:
        return None
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = _mp_pose.process(rgb)
    if not res.pose_landmarks:
        return None
    lm = res.pose_landmarks.landmark
    def xy(name):
        p = lm[_mp_idx[name]]
        return [float(p.x*w), float(p.y*h)]
    l_sh = xy("l_shoulder"); r_sh = xy("r_shoulder")
    neck = [(l_sh[0]+r_sh[0])/2.0, (l_sh[1]+r_sh[1])/2.0]
    sample = {
        "nose": xy("nose"), "neck": neck,
        "r_shoulder": r_sh, "r_elbow": xy("r_elbow"), "r_wrist": xy("r_wrist"),
        "l_shoulder": l_sh, "l_elbow": xy("l_elbow"), "l_wrist": xy("l_wrist"),
        "r_hip": xy("r_hip"), "r_knee": xy("r_knee"), "r_ankle": xy("r_ankle"),
        "l_hip": xy("l_hip"), "l_knee": xy("l_knee"), "l_ankle": xy("l_ankle"),
    }
    # Normalize by shoulder width
    shw = np.linalg.norm(np.array(r_sh) - np.array(l_sh)) or 1.0
    for k in list(sample.keys()):
        v = np.array(sample[k], dtype=float) / shw
        sample[k] = [float(v[0]), float(v[1])]
    return sample

def draw_skeleton(frame, sample):
    img = frame.copy()
    color = (0, 255, 0)
    def pt(k):
        v = sample[k]
        return int(v[0]*200), int(v[1]*200)
    pairs = [("neck","r_shoulder"),("r_shoulder","r_elbow"),("r_elbow","r_wrist"),
             ("neck","l_shoulder"),("l_shoulder","l_elbow"),("l_elbow","l_wrist"),
             ("neck","r_hip"),("r_hip","r_knee"),("r_knee","r_ankle"),
             ("neck","l_hip"),("l_hip","l_knee"),("l_knee","l_ankle")]
    for a,b in pairs:
        ax, ay = pt(a); bx, by = pt(b)
        cv2.line(img, (ax,ay), (bx,by), color, 2)
    for k in sample.keys():
        x,y = pt(k)
        cv2.circle(img, (x,y), 3, (0,0,255), -1)
    return img

REGION_MAP = {
    "shoulder": ["l_shoulder_", "r_shoulder_", "shoulder_", "trunk_"],
    "elbow": ["l_elbow_", "r_elbow_"],
    "knee": ["l_knee_", "r_knee_"],
    "ankle": ["l_ankle_", "r_ankle_"],
    "back": ["trunk_", "spine_", "hip_"]
}

def region_scores(FEATURES, reg_importance, xrow):
    contrib = {}
    for i, f in enumerate(FEATURES):
        val = float(xrow[i]); imp = float(reg_importance[i])
        if imp <= 0: continue
        for region, prefixes in REGION_MAP.items():
            if any(f.startswith(p) for p in prefixes):
                contrib[region] = contrib.get(region, 0.0) + imp * abs(val)
    if contrib:
        m = max(contrib.values()) or 1.0
        for k in contrib: contrib[k] = contrib[k] / m
    return contrib

def draw_hotspots(frame, sample, scores):
    img = frame.copy()
    anchors = {
        "shoulder": sample.get("neck") or sample.get("r_shoulder"),
        "elbow": sample.get("r_elbow"),
        "knee": sample.get("r_knee"),
        "ankle": sample.get("r_ankle"),
        "back": sample.get("r_hip")
    }
    def col(s):
        s = max(0.0, min(1.0, s))
        r = int(255*s)
        g = int(255*(1.0 - 0.5*s))
        return (0, g, r)
    for region, score in scores.items():
        pt = anchors.get(region)
        if not pt: continue
        x = int(pt[0]*200); y = int(pt[1]*200)
        c = col(score)
        cv2.circle(img, (x, y), 14, c, 3)
        cv2.putText(img, f"{region}:{int(score*100)}%", (x+8, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    return img

app = FastAPI(title="Cricket Biomechanics API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SCALER = joblib.load(MODELS / "scaler.pkl")
REG = XGBRegressor(); REG.load_model(str(MODELS / "xgb_performance.json"))
CLF = XGBClassifier(); CLF.load_model(str(MODELS / "xgb_injury.json"))
FEATURES = json.load(open(MODELS / "feature_list.json", "r"))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    name = file.filename.lower()
    content = await file.read()
    pose_json = None
    preview_b64 = None

    if name.endswith(".json"):
        pose_json = json.loads(content.decode("utf-8"))
    elif name.endswith((".mp4", ".mov", ".avi", ".mkv")):
        with NamedTemporaryFile(delete=False, suffix=name[name.rfind("."):]) as tmp:
            tmp.write(content); tmp.flush()
            video_path = tmp.name
        pose_json = infer_pose_from_video(video_path, out_json_path=None, target_fps=24)
        # Preview overlay from a frame near the middle
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        mid = max(0, total // 2 - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, frame = cap.read()
        if ok:
            sample = None
            for _ in range(20):
                sample = detect_one_frame(frame)
                if sample is not None: break
                ok, frame = cap.read()
                if not ok: break
            if sample is not None:
                feats_tmp = extract_features(pose_json)
                x_tmp = np.array([feats_tmp.get(f, 0.0) for f in FEATURES]).reshape(1, -1)
                xs_tmp = SCALER.transform(x_tmp)
                scores_tmp = region_scores(FEATURES, REG.feature_importances_, x_tmp[0])
                vis = draw_skeleton(frame, sample)
                vis = draw_hotspots(vis, sample, scores_tmp)
                ok2, jpg = cv2.imencode(".jpg", vis)
                if ok2:
                    preview_b64 = base64.b64encode(jpg.tobytes()).decode("utf-8")
        cap.release()
    else:
        raise HTTPException(status_code=400, detail="Upload a video (.mp4/.mov) or keypoints .json")

    feats = extract_features(pose_json)
    x = np.array([feats.get(f, 0.0) for f in FEATURES]).reshape(1, -1)
    xs = SCALER.transform(x)
    perf = float(REG.predict(xs)[0])
    inj = int(CLF.predict(xs)[0])
    reg_importance = REG.feature_importances_
    top_idx = np.argsort(reg_importance)[::-1][:5]
    top = [{FEATURES[i]: float(reg_importance[i]*x[0, i])} for i in top_idx]

    # hotspots for parity with live
    scores = region_scores(FEATURES, reg_importance, x[0])
    hot = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
    hot = [{"region": k, "score": float(v)} for k, v in hot]

    return {
        "performance_score": perf,
        "injury_risk": inj,
        "top_features": top,
        "hotspots": hot,
        "preview": preview_b64  # null for JSON uploads
    }

@app.websocket("/ws/realtime")
async def ws_realtime(ws: WebSocket):
    await ws.accept()
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = max(1, int(round(fps / 24)))
    buf = []
    miss = 0; total = 0
    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                await asyncio.sleep(0.02); continue
            if i % step != 0:
                i += 1; continue
            sample = detect_one_frame(frame)
            total += 1
            if sample is None:
                miss += 1
                await ws.send_json({"status": "no_pose", "miss_ratio": miss/total})
                i += 1; continue

            buf.append(sample)
            if len(buf) > 48:
                buf = buf[-48:]

            if len(buf) >= 24 and (len(buf) % 4 == 0):
                pose_json = {"joints": PROJECT_JOINTS, "frames": buf, "fps": 24.0}
                feats = extract_features(pose_json)
                x = np.array([feats.get(f, 0.0) for f in FEATURES]).reshape(1, -1)
                xs = SCALER.transform(x)
                perf = float(REG.predict(xs)[0])
                inj = int(CLF.predict(xs)[0])

                reg_importance = REG.feature_importances_
                scores = region_scores(FEATURES, reg_importance, x[0])
                vis = draw_skeleton(frame, sample)
                vis = draw_hotspots(vis, sample, scores)
                ok2, jpg = cv2.imencode(".jpg", vis)
                b64 = base64.b64encode(jpg.tobytes()).decode("utf-8") if ok2 else None
                hot = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
                hot = [{"region": k, "score": float(v)} for k, v in hot]

                await ws.send_json({
                    "performance_score": perf,
                    "injury_risk": inj,
                    "miss_ratio": miss/total,
                    "hotspots": hot,
                    "frame": b64
                })
            i += 1
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        pass
    finally:
        cap.release()
