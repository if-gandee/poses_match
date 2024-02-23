import networkx as nx
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from components import Pose, Match

class MatchingGraph:
    def __init__(self) -> None:
        self.source_pose_map : dict[int, Pose] = defaultdict(Pose)
        self.target_pose_map : dict[int, Pose] = defaultdict(Pose)
        self.node_map : dict[tuple[int, int], int] = defaultdict(int)
        self.source2targets_map : dict[int, list] = defaultdict(list[int])
        self.graph = nx.DiGraph()
        self.match_list : list[Match] = []
        self.window_type = None 
    
    def add_source_pose(self, pose : Pose | list[Pose]) :
        if isinstance(pose, Pose):
            self.source_pose_map[pose.uid] = pose
        elif isinstance(pose, list):
            for p in pose:
                self.source_pose_map[p.uid] = p
        else:
            raise ValueError(f"pose should be Pose or list of Pose, not {type(pose)}")
    
    def add_target_pose(self, pose: Pose | list[Pose]):
        if isinstance(pose, Pose):
            self.target_pose_map[pose.uid] = pose
        elif isinstance(pose, list):
            for p in pose:
                self.target_pose_map[p.uid] = p
        else:
            raise ValueError(f"pose should be Pose or list of Pose, not {type(pose)}")
    
    def add_match(self, source_pose_uid, target_pose_uid):
        source_pose = self.source_pose_map[source_pose_uid]
        target_pose = self.target_pose_map[target_pose_uid]
        match = Match(source_pose=source_pose, target_pose=target_pose)
        match.update_loss()
        self.match_list.append(match)
    
    def match_by_frame_index_window(self, id_window: int):
        self.match_list.clear()
        self.window_type = "frame_index"
        target_frame_index_to_uid_map = {pose.frame_index: pose.uid for pose in self.target_pose_map.values()}
        target_frame_index_list = list(target_frame_index_to_uid_map.keys())
        target_frame_index_list.sort()
        source_frame_index_to_uid_map = {pose.frame_index: pose.uid for pose in self.source_pose_map.values()}
        source_frame_index_list = list(source_frame_index_to_uid_map.keys())
        source_frame_index_list.sort()
        for source_frame_index in source_frame_index_list:
            target_frame_index_range = range(max(0, source_frame_index-id_window), source_frame_index+id_window)
            for target_frame_index in target_frame_index_range:
                if target_frame_index in target_frame_index_list:
                    self.add_match(source_frame_index_to_uid_map[source_frame_index], target_frame_index_to_uid_map[target_frame_index])
    
    def match_by_time_window(self, time_window_ms: float):
        self.match_list.clear()
        self.window_type = "timestamp"
        target_timestamp_to_uid_map = {pose.timestamp_ms: pose.uid for pose in self.target_pose_map.values()}
        target_timestamp_list = list(target_timestamp_to_uid_map.keys())
        target_timestamp_list.sort()
        source_timestamp_to_uid_map = {pose.timestamp_ms: pose.uid for pose in self.source_pose_map.values()}
        source_timestamp_list = list(source_timestamp_to_uid_map.keys())
        source_timestamp_list.sort()
        for source_timestamp in source_timestamp_list:
            candi_target_timestamp_list = [ts for ts in target_timestamp_list if abs(ts-source_timestamp) < time_window_ms]
            for target_timestamp in candi_target_timestamp_list:
                self.add_match(source_timestamp_to_uid_map[source_timestamp], target_timestamp_to_uid_map[target_timestamp])
    
    def build_graph(self):
        if len(self.match_list) == 0:
            raise ValueError("No match is added to the graph, please match first.")
        self.graph.clear()
        start_node = Match(source_pose=None, target_pose=None, loss=0.0)
        end_node = Match(source_pose=None, target_pose=None,loss=0.0)
        self.graph.add_node(0, match=start_node)
        for idx, candidate_match in enumerate(self.match_list):
            node_index = idx+1
            self.graph.add_node(node_index, match=candidate_match)
            self.node_map[(candidate_match.source_pose.uid, candidate_match.target_pose.uid)] = node_index
            self.source2targets_map[candidate_match.source_pose.uid].append(candidate_match.target_pose.uid)
        self.graph.add_node(len(self.match_list)+1, match=end_node)
        
        if self.window_type == "frame_index":
            source_frame_index_to_uid_map = {pose.frame_index: pose.uid for pose in self.source_pose_map.values()}
            source_frame_index_list = list(source_frame_index_to_uid_map.keys())
            source_frame_index_list.sort()
            sorted_source_uid_list = [source_frame_index_to_uid_map[frame_index] for frame_index in source_frame_index_list]
            target_frame_index_to_uid_map = {pose.frame_index: pose.uid for pose in self.target_pose_map.values()}
            target_frame_index_list = list(target_frame_index_to_uid_map.keys())
            target_frame_index_list.sort()
            sorted_target_uid_list = [target_frame_index_to_uid_map[frame_index] for frame_index in target_frame_index_list]
        elif self.window_type == "timestamp":
            source_timestamp_to_uid_map = {pose.timestamp_ms: pose.uid for pose in self.source_pose_map.values()}
            source_timestamp_list = list(source_timestamp_to_uid_map.keys())
            source_timestamp_list.sort()
            sorted_source_uid_list = [source_timestamp_to_uid_map[ts] for ts in source_timestamp_list]
            target_timestamp_to_uid_map = {pose.timestamp_ms: pose.uid for pose in self.target_pose_map.values()}
            target_timestamp_list = list(target_timestamp_to_uid_map.keys())
            target_timestamp_list.sort()
            sorted_target_uid_list = [target_timestamp_to_uid_map[ts] for ts in target_timestamp_list]
        else:
            raise ValueError("Window type should be either frame_index or timestamp")
        
        weighted_edges = []
        for target_id in self.source2targets_map[sorted_source_uid_list[0]]:
            weighted_edges.append((0, self.node_map[(sorted_source_uid_list[0], target_id)], 0.0))
        
        pbar = tqdm(total=len(sorted_source_uid_list)-2, desc="Building Graph")
        for idx, source_id in enumerate(sorted_source_uid_list[1:-1]):
            pbar.update(1)
            last_source_id = sorted_source_uid_list[idx]
            for target_id in self.source2targets_map[source_id]:
                current_node_num = self.node_map[(source_id, target_id)]
                for last_target_id in self.source2targets_map[last_source_id]:
                    if sorted_target_uid_list.index(last_target_id) < sorted_target_uid_list.index(target_id):
                        last_node_num = self.node_map[(last_source_id, last_target_id)]
                        last_node_loss = self.graph.nodes[last_node_num]["match"].loss
                        weighted_edges.append((last_node_num, current_node_num, last_node_loss))
                    else:
                        break
                        
        for target_id in self.source2targets_map[sorted_source_uid_list[-2]]:
            last_node_num = self.node_map[(sorted_source_uid_list[-2], target_id)]
            last_node_loss = self.graph.nodes[last_node_num]["match"].loss
            weighted_edges.append((last_node_num, len(self.match_list)+1, last_node_loss))
        # print(weighted_edges)
        self.graph.add_weighted_edges_from(weighted_edges)
    
    def find_shortest_path(self):
        # minWPath = nx.bellman_ford_path(self.graph, source=0, target=len(self.match_list)+1)
        min_path = nx.dijkstra_path(self.graph, source=0, target=len(self.match_list)+1)
        path_length = 0
        for idx in range(len(min_path)):
            path_length += self.graph.nodes[min_path[idx]]["match"].loss
        return min_path, path_length

    def solve(self):
        self.build_graph()
        print("Solving")
        minWPath, total_loss = self.find_shortest_path()
        matched_pairs = []
        for node_num in minWPath[1:-1]:
            matched_pairs.append((self.graph.nodes[node_num]['match'].source_pose, self.graph.nodes[node_num]['match'].target_pose))
        print("Solved")
        return matched_pairs, total_loss

if __name__ == "__main__":
    pass