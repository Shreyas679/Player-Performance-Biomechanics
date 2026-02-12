# ml/synthetic_data.py
import os, cv2, json, math, random
import numpy as np
from pathlib import Path

FPS = 24
CLIP_LEN = 48  # 2 sec
RES = (640, 360)
JOINTS = ["nose","neck","r_shoulder","r_elbow","r_wrist","l_shoulder","l_elbow","l_wrist",
          "r_hip","r_knee","r_ankle","l_hip","l_knee","l_ankle"]
EDGES = [("neck","r_shoulder"),("r_shoulder","r_elbow"),("r_elbow","r_wrist"),
         ("neck","l_shoulder"),("l_shoulder","l_elbow"),("l_elbow","l_wrist"),
         ("neck","r_hip"),("r_hip","r_knee"),("r_knee","r_ankle"),
         ("neck","l_hip"),("l_hip","l_knee"),("l_knee","l_ankle")]

def bowler_skeleton(t, variant):
    # Parametric 2D kinematics for a right-arm bowler delivering the ball.
    # t in [0,1]
    base_x, base_y = 320, 220
    stride = 60 + 30*variant["stride_amp"]
    trunk_flex = math.radians(10 + 15*variant["trunk_flex"])
    shoulder_rot = math.radians(70 + 30*variant["shoulder_rot"])
    elbow_flex = math.radians(40 + 30*variant["elbow_flex"])
    knee_valgus = math.radians(variant["knee_valgus_deg"])  # deviation

    # Hip oscillation for run-up to gather
    hip_shift = 10*math.sin(2*math.pi*(t*2))
    neck = np.array([base_x+hip_shift, base_y-60])
    r_hip = np.array([base_x+hip_shift+stride*t, base_y])
    l_hip = np.array([base_x+hip_shift-8, base_y])

    # Knees and ankles with simple leg chain and valgus deviation
    def leg_chain(hip, side=1):
        knee = hip + np.array([10*side + 5*math.sin(2*math.pi*t), 30])
        knee += np.array([10*math.sin(2*math.pi*t), 0])
        # apply valgus on front (right) knee
        if side==1:
            R = np.array([[math.cos(knee_valgus), -math.sin(knee_valgus)],
                          [math.sin(knee_valgus),  math.cos(knee_valgus)]])
            knee = hip + R @ (knee - hip)
        ankle = knee + np.array([5*side, 30])
        return knee, ankle
    r_knee, r_ankle = leg_chain(r_hip, side=1)
    l_knee, l_ankle = leg_chain(l_hip, side=-1)

    # Shoulders
    r_shoulder = neck + np.array([15, 5])
    l_shoulder = neck + np.array([-15, 5])

    # Trunk forward lean
    neck = neck + np.array([0, 5*variant["trunk_flex"]])

    # Bowling arm kinematics (right arm)
    arm_phase = t
    # upper arm angle
    ua = -math.pi/2 + shoulder_rot*math.sin(2*math.pi*arm_phase)
    # forearm with elbow flex
    fa = ua + (math.pi/2 - elbow_flex) + 0.2*math.sin(2*math.pi*arm_phase)

    upper_len, fore_len = 30, 25
    r_elbow = r_shoulder + np.array([upper_len*math.cos(ua), upper_len*math.sin(ua)])
    r_wrist = r_elbow + np.array([fore_len*math.cos(fa), fore_len*math.sin(fa)])

    # Left arm counterbalance
    la = math.pi/2 - 0.6*shoulder_rot*math.sin(2*math.pi*arm_phase)
    le = l_shoulder + np.array([upper_len*math.cos(la), upper_len*math.sin(la)])
    lw = le + np.array([fore_len*math.cos(la+0.4), fore_len*math.sin(la+0.4)])

    # Head and face
    nose = neck + np.array([0, -15])

    pts = {
        "nose":nose, "neck":neck,
        "r_shoulder":r_shoulder,"r_elbow":r_elbow,"r_wrist":r_wrist,
        "l_shoulder":l_shoulder,"l_elbow":le,"l_wrist":lw,
        "r_hip":r_hip,"r_knee":r_knee,"r_ankle":r_ankle,
        "l_hip":l_hip,"l_knee":l_knee,"l_ankle":l_ankle
    }
    return {k: [float(v[0]), float(v[1])] for k,v in pts.items()}

def render_frame(kp):
    img = np.ones((RES[1], RES[0], 3), np.uint8)*255
    # draw lines
    for a,b in EDGES:
        ax, ay = map(int, kp[a]); bx, by = map(int, kp[b])
        cv2.line(img, (ax,ay), (bx,by), (0,0,0), 2)
    # draw joints
    for k,(x,y) in kp.items():
        cv2.circle(img, (int(x), int(y)), 3, (0,0,255), -1)
    return img

def label_from_variant(variant):
    # Heuristic performance and injury risk
    perf = 80
    perf -= 10*variant["trunk_flex"]
    perf -= 8*abs(variant["knee_valgus_deg"])/10
    perf += 5*variant["shoulder_rot"]
    perf -= 6*variant["elbow_flex"]
    perf = max(0, min(100, perf + random.uniform(-3,3)))
    injury = 1 if (variant["trunk_flex"]>0.8 or abs(variant["knee_valgus_deg"])>12 or variant["shoulder_rot"]>0.9) else 0
    return float(perf), int(injury)

def gen_variant():
    return {
        "stride_amp": random.uniform(0.3, 1.0),
        "trunk_flex": random.uniform(0.0, 1.2),
        "shoulder_rot": random.uniform(0.4, 1.1),
        "elbow_flex": random.uniform(0.1, 1.0),
        "knee_valgus_deg": random.uniform(-15, 18),
    }

def generate_dataset(root=".", n=120):
    root = Path(root)
    (root/"data/raw_videos").mkdir(parents=True, exist_ok=True)
    (root/"data/keypoints").mkdir(parents=True, exist_ok=True)
    meta = []
    for i in range(n):
        variant = gen_variant()
        perf, injury = label_from_variant(variant)
        clip_id = f"clip_{i:04d}"
        # Pose frames
        frames = []
        for f in range(CLIP_LEN):
            t = f/(CLIP_LEN-1)
            kp = bowler_skeleton(t, variant)
            frames.append(kp)
        # Save keypoints JSON
        with open(root/"data/keypoints"/f"{clip_id}.json","w") as fjson:
            json.dump({"joints":JOINTS, "frames":frames, "fps":FPS}, fjson)
        # Render synthetic video
        outv = cv2.VideoWriter(str(root/"data/raw_videos"/f"{clip_id}.mp4"),
                               cv2.VideoWriter_fourcc(*"mp4v"), FPS, RES)
        for kp in frames:
            outv.write(render_frame(kp))
        outv.release()
        meta.append({"clip_id":clip_id,"performance_score":perf,"injury_risk":injury})
    # Save labels
    import pandas as pd
    pd.DataFrame(meta).to_csv(root/"data/labels.csv", index=False)
    print(f"Generated {n} clips")
if __name__ == "__main__":
    generate_dataset(".", n=160)
