import numpy as np
from typing import ClassVar, Callable

from attrs import define, field

from utils import find_transform_similarity


@define(kw_only=True, eq=False)
class Pose:
    UidPool : ClassVar[int] = 0
    
    uid : int = field(default=None)
    timestamp_ms : float | None = field(default=None)
    frame_index : int = field()
    keypoints_xy : np.ndarray  = field()    # shape (n, 2)
    bones : np.ndarray | None = field(default=None)
    keypoint_confidence : np.ndarray = field(default=None)   # len(keypoints) == len(keypoint_confidence), invalid confidence <= 0
    
    def __attrs_post_init__(self):
        if self.uid is None:
            self.uid = self.UidPool
            Pose.UidPool += 1
            
        if self.keypoint_confidence is None:
            self.keypoint_confidence = np.ones(len(self.keypoints_xy), dtype=float)
        else:
            assert len(self.keypoints_xy) == len(self.keypoint_confidence), "keypoints and keypoint_confidence should have same length"
    
@define(kw_only=True, eq=False)
class Match:
    source_pose : Pose | None = field(default=None)
    target_pose : Pose | None = field(default=None)
    transform : np.ndarray | None = field(default=None)
    loss : float | None = field(default=None)
    loss_function : Callable[[Pose, Pose], float] | None = field(default=None)
    
    def update_transform(self):
        shared_source_kps, shared_target_kps = self.get_shared_keypoints(self.source_pose, self.target_pose)
        self.transform = find_transform_similarity(shared_source_kps, shared_target_kps)
        return self.transform
    
    def update_loss(self):
        if self.loss_function is not None:
            self.loss = self.loss_function(self.source_pose, self.target_pose)
            return self.loss
        if self.transform is None:
            self.update_transform()
        if self.source_pose.bones is not None and self.target_pose.bones is not None:
            shared_bones = self.get_shared_bones(self.source_pose, self.target_pose)
            kps_1 = self.source_pose.keypoints_xy @ self.transform[:2, :2] + self.transform[2,:2]
            kps_2 = self.target_pose.keypoints_xy
            self.loss = self.cosine_distance_with_bones(kps_1, kps_2, shared_bones)
            return self.loss
        shared_source_kps, shared_target_kps = self.get_shared_keypoints(self.source_pose, self.target_pose)
        shared_source_kps = shared_source_kps @ self.transform[:2, :2] + self.transform[2,:2]
        self.loss = self.cosine_distance(shared_source_kps, shared_target_kps)
        return self.loss
    
    @staticmethod
    def get_shared_bones(pose_1: Pose, pose_2: Pose):
        valid_mask = (pose_1.keypoint_confidence > 0) & (pose_2.keypoint_confidence > 0)
        shared_bones = []
        for bone in pose_1.bones:
            if valid_mask[bone[0]] and valid_mask[bone[1]]:
                shared_bones.append(bone)
        return shared_bones
    
    @staticmethod
    def get_shared_keypoints(pose_1: Pose, pose_2 : Pose):
        valid_mask = (pose_1.keypoint_confidence > 0) & (pose_2.keypoint_confidence > 0)
        kps_1 = pose_1.keypoints_xy[valid_mask]
        kps_2 = pose_2.keypoints_xy[valid_mask]
        return kps_1, kps_2
        
    @staticmethod
    def cosine_distance(kps1, kps2):
        kps1 = kps1.flatten()
        kps2 = kps2.flatten()
        cossim = kps1.dot(np.transpose(kps2)) / (np.linalg.norm(kps1) * np.linalg.norm(kps2))
        cosdist = abs(1 - cossim)
        cosdist = cosdist/len(kps1)
        return cosdist

    @staticmethod
    def cosine_distance_with_bones(kps_1, kps_2, bones):
        total_cosdist = 0
        for bone in bones:
            bone_vec_1 = kps_1[bone[1]] - kps_1[bone[0]]
            bone_vec_2 = kps_2[bone[1]] - kps_2[bone[0]]
            dist = (np.linalg.norm(bone_vec_1) * np.linalg.norm(bone_vec_2))
            if dist == 0:
                cossim = 1
            else:
                cossim = bone_vec_1.dot(np.transpose(bone_vec_2)) / dist
            cosdist = abs(1 - cossim)
            total_cosdist += cosdist
        return total_cosdist + (13-len(bones)) * 0.5
    
    
if __name__ == "__main__":
    import random
    import cv2
    bones = [
            [2, 3], [2, 6], [3, 4], [4, 5], 
            [6, 7], [7, 8], [2, 9], [9, 10], 
            [10, 11], [2, 12], [12, 13], [13, 14], 
            [2, 1],
        ]
    bones = np.array(bones, dtype=np.uint8)
    bones = bones - 1
    bones = None
    kps1 = np.load("poses_seq\\111\\00226.npy")
    kps2 = np.load("poses_seq\\111\\00226.npy")
    keypoint_confidence1 = np.ones(len(kps1), dtype=np.float32)
    keypoint_confidence1[kps1[:, 0] < 0] = 0
    keypoint_confidence2 = np.ones(len(kps2), dtype=np.float32)
    keypoint_confidence2[kps2[:, 0] < 0] = 0
    rot_mat = cv2.getRotationMatrix2D((0.5, 0.5), 10, 1)
    print(rot_mat)
    kps2 = kps2 @ rot_mat[:2, :2] + rot_mat[:2, 2]
    frame1 = Pose(keypoints_xy=kps1, frame_index=26, bones=bones, keypoint_confidence=keypoint_confidence1)
    frame2 = Pose(keypoints_xy=kps2, frame_index=26, bones=bones, keypoint_confidence=keypoint_confidence2)
    cur_match = Match(source_pose=frame1, target_pose=frame2)
    loss = cur_match.update_loss()
    print(loss)
    
