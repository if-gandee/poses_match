import cv2
import os, sys
import numpy as np
from glob import glob
from controlnet_aux import OpenposeDetector
from controlnet_aux.open_pose import draw_poses


open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
open_pose = open_pose.to("cuda") 

def detect_keypoints(image, detect_resolution):
    if image.shape[0] != detect_resolution or image.shape[1] != detect_resolution:
        detect_image = cv2.resize(image, (detect_resolution, detect_resolution))
    detect_image = cv2.cvtColor(detect_image, cv2.COLOR_BGR2RGB)
    detect_res = open_pose.detect_poses(oriImg=detect_image, include_face=False, include_hand=False)
    final_keypoints = None
    area = 0
    final_res = None
    if detect_res is not None and len(detect_res) > 0:
        for res in detect_res:
            keypoints = np.ones((18, 2), dtype=np.float32) * -1
            raw_keypoints = res.body.keypoints
            for idx, kp in enumerate(raw_keypoints):
                if kp is None:
                    continue
                keypoints[idx, 0] = kp.x
                keypoints[idx, 1] = kp.y
            #caclulate the area of the bounding box
            
            rect = cv2.boundingRect(keypoints[keypoints>=0].reshape(-1, 2)*512)
            if area < rect[2]*rect[3]:
                area = rect[2]*rect[3]
                final_keypoints = keypoints
                final_res = res
    return final_keypoints, final_res


vid_name = "kemu3-1"
video_cap = cv2.VideoCapture("kemu3-1.mp4")

idx = 0
while True:
    ret, frame = video_cap.read()
    if not ret:
        break
    keypoints1, pose_res = detect_keypoints(frame, 512)
    np.save(f"poses_seq\\{vid_name}\\{idx:05}.npy", keypoints1)
    pose_map = draw_poses([pose_res], 512, 512, draw_body=True, draw_hand=False, draw_face=False)
    cv2.imwrite(f"poses_seq\\{vid_name}\\{idx:05}_pose.png", pose_map)
    cv2.imwrite(f"poses_seq\\{vid_name}\\{idx:05}_ori.png", frame)
    idx += 1
    print(idx)
    

vid_name = "kemu3-2"
video_cap = cv2.VideoCapture("kemu3-2.mp4")

idx = 0
while True:
    ret, frame = video_cap.read()
    if not ret:
        break
    keypoints1, pose_res = detect_keypoints(frame, 512)
    np.save(f"poses_seq\\{vid_name}\\{idx:05}.npy", keypoints1)
    pose_map = draw_poses([pose_res], 512, 512, draw_body=True, draw_hand=False, draw_face=False)
    cv2.imwrite(f"poses_seq\\{vid_name}\\{idx:05}_pose.png", pose_map)
    cv2.imwrite(f"poses_seq\\{vid_name}\\{idx:05}_ori.png", frame)
    idx += 1
    print(idx)