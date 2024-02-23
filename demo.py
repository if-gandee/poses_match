import cv2
import os
import numpy as np
import distinctipy
import math

from components import Pose
from graph import MatchingGraph


def draw_bodypose(canvas: np.ndarray, keypoints: np.ndarray, bones:np.ndarray) -> np.ndarray:

    H, W, C = canvas.shape
    H, W, C = 1,1,1
    stickwidth = 4

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for (k1_index, k2_index), color in zip(bones, colors):
        keypoint1 = keypoints[k1_index]
        keypoint2 = keypoints[k2_index]

        if keypoint1 is None or keypoint2 is None or keypoint1[0] < 0 or keypoint2[0] < 0:
            continue

        Y = np.array([keypoint1[0], keypoint2[0]]) * float(W)
        X = np.array([keypoint1[1], keypoint2[1]]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None or keypoint[0] < 0:
            continue

        x, y = keypoint
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

    return canvas


bones = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1]
    ]
bones = np.array(bones, dtype=np.uint8)
bones = bones - 1

matching_graph = MatchingGraph()
end_frame_idx = 2000
ref_vid_name = "222"
cur_vid_name = "444"
output_dir = "outputs\\output13"
os.makedirs(output_dir, exist_ok=True)
source_pose_list = []
target_pose_list = []

source_key_frames = [0, 78, 91, 117, 133, 156, 176, 184, 213, 223, 233, 263, 280, 297, 304, 314,
                     334, 340, 352, 365, 371, 384, 396, 404, 414, 420, 429, 440, 446, 459, 481, 490,
                     500, 533, 546, 556, 566, 577, 601, 609, 622, 649, 663, 674, 684, 700 ]

#source pose，用于参考的源关键帧，建议手动设置，也可以间隔一定帧数采样，根据动作复杂程度和重复情况决定采样间隔，过大和过小都会影响匹配效果
for i in range(0, end_frame_idx, 30):
# for i in source_key_frames:
    ref_frame_idx = i
    #时间戳，单位为毫秒，可以根据帧数计算，也可以使用真实的时间戳。
    timestamp_ms = ref_frame_idx * 1000 / 30
    #关键点，可以使用任意方法检测到的2d关键点。本例中使用提前保存好的Openpose检测的关键点
    keypoints = np.load(f"poses_seq\\{ref_vid_name}\\{ref_frame_idx+350:05}.npy")  #shape of keypoints is (n, 2)
    #openpose检测的关键点是归一化的，是否归一化不影响匹配效果，但是在绘制关键点时需要根据图像的实际尺寸进行缩放
    #为了便于绘制pose，这里将关键点坐标缩放到512*512的图像上
    keypoints = keypoints*512
    #关键点置信度，当且仅当置信度为0时，关键点被认为是无效的。请确保关键点和置信度的长度相同和置信度的准确性，否则会影响匹配效果。
    keypoint_confidence = np.ones(len(keypoints), dtype=np.float32)
    keypoint_confidence[keypoints[:, 0] < 0] = 0
    #创建Pose对象，其中bones是骨骼，可以不传，使用骨骼和不使用骨骼信息会影响loss计算
    pose = Pose(frame_index=ref_frame_idx, keypoints_xy=keypoints, keypoint_confidence=keypoint_confidence, 
                timestamp_ms=timestamp_ms, bones=bones)
    source_pose_list.append(pose)
    
#target pose，用于匹配的目标关键帧
for j in range(0, end_frame_idx+10):
    cur_frame_idx = j
    timestamp_ms = cur_frame_idx * 1000 / 30
    keypoints = np.load(f"poses_seq\\{cur_vid_name}\\{cur_frame_idx:05}.npy")
    keypoints = keypoints*512
    keypoint_confidence = np.ones(len(keypoints), dtype=np.float32)
    keypoint_confidence[keypoints[:, 0] < 0] = 0
    pose = Pose(frame_index=cur_frame_idx, keypoints_xy=keypoints, keypoint_confidence=keypoint_confidence,
                timestamp_ms=timestamp_ms, bones=bones)
    target_pose_list.append(pose)

#创建完source_pose_list和target_pose_list后，调用MatchingGraph的add_source_pose和add_target_pose方法添加到匹配图中
matching_graph.add_source_pose(source_pose_list)
matching_graph.add_target_pose(target_pose_list)

#调用match_by_frame_index_window或match_by_time_window方法，将根据帧号或时间戳创建有向权重图，用于全局优化。这两个方法二选其一即可。
#参数的含义是搜索半径，会从参考帧的前后若干帧或若干时间中搜索最佳匹配，搜索半径越大，计算量越大。
#搜索半径越大并非一定能得到更好的匹配结果，特别是当存在较多重复动作的情况。需要根据实际情况调整。
matching_graph.match_by_frame_index_window(30)    #search ridus is 40
# matching_graph.match_by_time_window(1000)

#完成建图之后，调用solve方法求解最优匹配，返回匹配结果和总损失。matched_pairs为[(source_pose, target_pose), ...]的列表，total_loss为float类型的总损失。
matched_pairs, total_loss = matching_graph.solve()
print(f"Total loss: {total_loss}")
#至此，匹配完成，matched_pairs为匹配结果，total_loss为总损失。可以根据匹配结果进行后续处理，如可视化，姿态迁移等。


#以下代码用于可视化匹配结果，将匹配结果保存为视频和图片
video_writer = cv2.VideoWriter(os.path.join(output_dir, "output.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 30, (512*3, 512*2))
last_source_pose, last_target_pose = matched_pairs[0]
last_source_frame_idx = last_source_pose.frame_index
last_target_frame_idx = last_target_pose.frame_index
for source_pose, target_pose in matched_pairs[1:]:
    matched_source_frame_idx = source_pose.frame_index
    matched_target_frame_idx = target_pose.frame_index
    matched_source_frame = cv2.imread(f"poses_seq\\{ref_vid_name}\\{last_source_frame_idx:05}_ori.png")
    matched_target_frame = cv2.imread(f"poses_seq\\{cur_vid_name}\\{(last_target_frame_idx):05}_ori.png")
    matched_source_frame = cv2.resize(matched_source_frame, (512, 512))
    matched_target_frame = cv2.resize(matched_target_frame, (512, 512))
    matched_source_pose_map = draw_bodypose(matched_source_frame.copy(), last_source_pose.keypoints_xy, bones)
    matched_target_pose_map = draw_bodypose(matched_target_frame.copy(), last_target_pose.keypoints_xy, bones)
    
    
    for src_idx in range(last_source_frame_idx, matched_source_frame_idx):
        source_img = cv2.imread(f"poses_seq\\{ref_vid_name}\\{src_idx:05}_ori.png")
        target_img_ori = cv2.imread(f"poses_seq\\{cur_vid_name}\\{(src_idx):05}_ori.png")
        source_img = cv2.resize(source_img, (512, 512))
        target_img_ori = cv2.resize(target_img_ori, (512, 512))
        
        target_ori_pose = np.load(f"poses_seq\\{cur_vid_name}\\{src_idx:05}.npy")*512
        target_img_ori_with_pose = draw_bodypose(target_img_ori.copy(), target_ori_pose, bones)
        
        source_percentage = (src_idx - last_source_frame_idx) / (matched_source_frame_idx - last_source_frame_idx)
        target_float_frame_idx = last_target_frame_idx + (matched_target_frame_idx - last_target_frame_idx) * source_percentage
        target_int_frame_idx_1 = int(target_float_frame_idx)
        target_int_frame_idx_2 = target_int_frame_idx_1 + 1
        target_img_1 = cv2.imread(f"poses_seq\\{cur_vid_name}\\{(target_int_frame_idx_1):05}_ori.png")
        target_img_2 = cv2.imread(f"poses_seq\\{cur_vid_name}\\{(target_int_frame_idx_2):05}_ori.png")
        target_percentage = target_float_frame_idx - target_int_frame_idx_1
        target_img = (1-target_percentage) * target_img_1 + target_percentage * target_img_2
        target_img = cv2.resize(target_img, (512, 512))
        target_img = target_img.astype(np.uint8)
        
        
        cv2.putText(source_img, f"source video", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(target_img_ori, f"target video", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(target_img, f"aligned target video", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        row1 = cv2.hconcat([source_img, target_img_ori, target_img])
        row2 = cv2.hconcat([matched_source_pose_map, target_img_ori_with_pose, matched_target_pose_map])
        combined_img = cv2.vconcat([row1, row2])
        combined_img = cv2.resize(combined_img, (512*3, 512*2))
        video_writer.write(combined_img)
        cv2.imwrite(os.path.join(output_dir, f"{src_idx:05}.png"), combined_img)
        cv2.imshow("combined", combined_img)
        cv2.waitKey(1)
    last_source_pose = source_pose
    last_target_pose = target_pose
    last_source_frame_idx = matched_source_frame_idx
    last_target_frame_idx = matched_target_frame_idx
cv2.destroyAllWindows()
video_writer.release()
