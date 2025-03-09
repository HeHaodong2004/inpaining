import time
import torch
import numpy as np
from rl_utils import *
from rl_parameter import *
from utils import quads
import matplotlib.pyplot as plt



class GroundTruthNodeManager:
    def __init__(self, node_manager, ground_truth_map_info, device='cpu', plot=False):
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.node_manager = node_manager
        self.ground_truth_map_info = ground_truth_map_info
        self.ground_truth_node_coords = None
        self.ground_truth_node_utility = None
        self.explored_sign = None
        self.device = device
        self.plot = plot

        self.initialize_graph()

    def get_ground_truth_observation(self, robot_location):
        self.update_graph()

        all_node_coords = []
        # 从 belief 中读取节点
        for node in self.node_manager.nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
        # 还将未探索的 ground truth 节点加入
        for node in self.nodes_dict.__iter__():
            if node.data.explored == 0:
                all_node_coords.append(node.data.coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = []
        explored_sign = []
        guidepost = []

        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j

        # 计算当前从 robot_location 出发的最短路径（基于 utility > 0 节点）
        shortest_path = self.compute_shortest_path(robot_location)
        # 输出调试信息（可选）
        # print("Debug: computed shortest path:", shortest_path)

        for i, coords in enumerate(all_node_coords):
            node = self.nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
            explored_sign.append(node.explored)
            # 如果当前节点坐标出现在最短路径中，则标记为 guidepost
            if (coords[0], coords[1]) in shortest_path:
                guidepost.append(1)
            else:
                guidepost.append(0)
            for neighbor in node.neighbor_list:
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                index = index[0][0]
                adjacent_matrix[i, index] = 0

        utility = np.array(utility)
        explored_sign = np.array(explored_sign)
        guidepost = np.array(guidepost)

        current_index = np.argwhere(node_coords_to_check == robot_location[0] + robot_location[1] * 1j)[0][0]

        neighbor_indices = []
        current_node_in_belief = self.node_manager.nodes_dict.find(robot_location.tolist()).data
        for neighbor in current_node_in_belief.neighbor_list:
            index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)[0][0]
            neighbor_indices.append(index)
        neighbor_indices = np.sort(np.array(neighbor_indices))

        self.ground_truth_node_coords = all_node_coords
        self.ground_truth_node_utility = utility
        self.explored_sign = explored_sign

        node_coords = all_node_coords
        node_utility = utility.reshape(-1, 1)
        # 这里将 explored_sign 和 guidepost 分别作为两个特征
        node_explored = explored_sign.reshape(-1, 1)
        node_guidepost = guidepost.reshape(-1, 1)
        # 额外添加 uncertainty 特征
        if self.ground_truth_map_info.uncertainty is None:
            node_uncertainty = np.zeros((n_nodes, 1))
        else:
            node_uncertainty = []
            for coords in all_node_coords:
                cell = get_cell_position_from_coords(coords, self.ground_truth_map_info)
                unc = self.ground_truth_map_info.uncertainty[cell[1], cell[0]]
                node_uncertainty.append(unc)
            node_uncertainty = np.array(node_uncertainty).reshape(-1, 1)

        # 拼接后的特征维度（2 + 1 + 1 + 2 = 6）
        node_inputs = np.concatenate((node_coords, node_utility, node_guidepost, node_uncertainty), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        assert node_coords.shape[0] < NODE_PADDING_SIZE, print(node_coords.shape[0], NODE_PADDING_SIZE)
        padding = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_nodes))
        node_inputs = padding(node_inputs)

        node_padding_mask = torch.zeros((1, 1, n_nodes), dtype=torch.int16).to(self.device)
        node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_nodes), dtype=torch.int16).to(self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        edge_mask = torch.tensor(adjacent_matrix).unsqueeze(0).to(self.device)
        padding = torch.nn.ConstantPad2d((0, NODE_PADDING_SIZE - n_nodes, 0, NODE_PADDING_SIZE - n_nodes), 1)
        edge_mask = padding(edge_mask)

        current_in_edge = np.argwhere(neighbor_indices == current_index)[0][0]
        current_edge = torch.tensor(neighbor_indices).unsqueeze(0).to(self.device)
        k_size = current_edge.size()[-1]
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
        current_edge = padding(current_edge)
        current_edge = current_edge.unsqueeze(-1)

        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)
        edge_padding_mask = padding(edge_padding_mask)

        current_index = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)

        return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask]

    # 新增：根据 robot_location 和当前 nodes_dict 计算最短路径（不包含起点），路径仅考虑 utility > 0 的节点
    def compute_shortest_path(self, robot_location):
        dist, prev = self.Dijkstra(robot_location)
        best_dist = float('inf')
        best_node = None
        for node in self.nodes_dict.__iter__():
            node_data = node.data
            if node_data.utility > 0:
                key = (node_data.coords[0], node_data.coords[1])
                if key == (robot_location[0], robot_location[1]):
                    continue
                if key in dist and dist[key] < best_dist:
                    best_dist = dist[key]
                    best_node = key
        if best_node is None:
            return []
        path, _ = self.get_Dijkstra_path_and_dist(dist, prev, best_node)
        return path

    def add_node_to_dict(self, coords):
        key = (coords[0], coords[1])
        node = Node(coords)
        self.nodes_dict.insert(point=key, data=node)
        return node

    def initialize_graph(self):
        node_coords = self.get_ground_truth_node_coords(self.ground_truth_map_info)
        for coords in node_coords:
            self.add_node_to_dict(coords)

        for node in self.nodes_dict.__iter__():
            node.data.get_neighbor_nodes(self.ground_truth_map_info, self.nodes_dict)

    def update_graph(self):
        for node in self.node_manager.nodes_dict.__iter__():
            coords = node.data.coords
            ground_truth_node = self.nodes_dict.find(coords.tolist())
            if ground_truth_node:
                ground_truth_node.data.utility = node.data.utility
                ground_truth_node.data.explored = 1
                ground_truth_node.data.visited = node.data.visited
            else:
                #print('Warning: Node in belief not found in prediction')
                self.add_node_to_dict(coords)

    def get_ground_truth_node_coords(self, ground_truth_map_info):
        x_min = ground_truth_map_info.map_origin_x
        y_min = ground_truth_map_info.map_origin_y
        x_max = ground_truth_map_info.map_origin_x + (ground_truth_map_info.map.shape[1] - 1) * CELL_SIZE
        y_max = ground_truth_map_info.map_origin_y + (ground_truth_map_info.map.shape[0] - 1) * CELL_SIZE

        if x_min % NODE_RESOLUTION != 0:
            x_min = (x_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
        if x_max % NODE_RESOLUTION != 0:
            x_max = x_max // NODE_RESOLUTION * NODE_RESOLUTION
        if y_min % NODE_RESOLUTION != 0:
            y_min = (y_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
        if y_max % NODE_RESOLUTION != 0:
            y_max = y_max // NODE_RESOLUTION * NODE_RESOLUTION

        x_coords = np.arange(x_min, x_max + 0.1, NODE_RESOLUTION)
        y_coords = np.arange(y_min, y_max + 0.1, NODE_RESOLUTION)
        t1, t2 = np.meshgrid(x_coords, y_coords)
        nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        nodes = np.around(nodes, 1)

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, ground_truth_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < ground_truth_map_info.map.shape[0] and 0 <= cell[0] < ground_truth_map_info.map.shape[1]
            if ground_truth_map_info.map[cell[1], cell[0]] == FREE:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

        return nodes

    def Dijkstra(self, start):
        q = set()
        dist_dict = {}
        prev_dict = {}

        for node in self.nodes_dict.__iter__():
            coords = node.data.coords
            key = (coords[0], coords[1])
            dist_dict[key] = 1e8
            prev_dict[key] = None
            q.add(key)

        assert (start[0], start[1]) in dist_dict.keys()
        dist_dict[(start[0], start[1])] = 0

        while len(q) > 0:
            u = None
            for coords in q:
                if u is None:
                    u = coords
                elif dist_dict[coords] < dist_dict[u]:
                    u = coords

            q.remove(u)

            node = self.nodes_dict.find(u).data
            for neighbor_node_coords in node.neighbor_list:
                v = (neighbor_node_coords[0], neighbor_node_coords[1])
                if v in q:
                    cost = ((neighbor_node_coords[0] - u[0]) ** 2 + (neighbor_node_coords[1] - u[1]) ** 2) ** (1 / 2)
                    cost = np.round(cost, 2)
                    alt = dist_dict[u] + cost
                    if alt < dist_dict[v]:
                        dist_dict[v] = alt
                        prev_dict[v] = u

        return dist_dict, prev_dict

    def get_Dijkstra_path_and_dist(self, dist_dict, prev_dict, end):
        if (end[0], end[1]) not in dist_dict:
            return [], 1e8

        d = dist_dict[(end[0], end[1])]
        path = [(end[0], end[1])]
        prev = prev_dict[(end[0], end[1])]
        while prev is not None and prev != (end[0], end[1]):
            path.append(prev)
            prev = prev_dict[prev]
        path.reverse()
        return path[1:], np.round(d, 2)

    def plot_ground_truth_env(self, robot_location, coverage_path=None):
        plt.subplot(1, 3, 3)
        plt.imshow(self.ground_truth_map_info.map, cmap='gray')
        plt.axis('off')
        robot = get_cell_position_from_coords(robot_location, self.ground_truth_map_info)
        nodes = get_cell_position_from_coords(self.ground_truth_node_coords, self.ground_truth_map_info)
        plt.imshow(self.ground_truth_map_info.map, cmap='gray')
        plt.scatter(nodes[:, 0], nodes[:, 1], c=self.explored_sign, zorder=2)
        plt.plot(robot[0], robot[1], 'mo', markersize=16, zorder=5)
        if coverage_path is not None:
            path_cell = get_cell_position_from_coords(np.array(coverage_path), self.ground_truth_map_info)
            plt.plot(path_cell[:, 0], path_cell[:, 1], 'b', linewidth=2, zorder=1)

    # 新增：根据 robot_location 计算从当前节点图中 utility > 0 的节点中最近目标的最短路径（不包含起点）
    def compute_shortest_path(self, robot_location):
        dist, prev = self.Dijkstra(robot_location)
        best_dist = float('inf')
        best_node = None
        for node in self.nodes_dict.__iter__():
            node_data = node.data
            if node_data.utility > 0:
                key = (node_data.coords[0], node_data.coords[1])
                if key == (robot_location[0], robot_location[1]):
                    continue
                if key in dist and dist[key] < best_dist:
                    best_dist = dist[key]
                    best_node = key
        if best_node is None:
            return []
        path, _ = self.get_Dijkstra_path_and_dist(dist, prev, best_node)
        return path



class Node:
    def __init__(self, coords):
        self.coords = coords
        self.utility = -(SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        self.explored = 0
        self.visited = 0

        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_list = []
        self.neighbor_list.append(self.coords)

    def get_neighbor_nodes(self, ground_truth_map_info, nodes_dict):
        center_index = self.neighbor_matrix.shape[0] // 2
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                else:
                    if i == center_index and j == center_index:
                        self.neighbor_matrix[i, j] = 1
                        continue

                    neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                                                          self.coords[1] + (j - center_index) * NODE_RESOLUTION]), 1)
                    neighbor_node = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                    if neighbor_node is None:
                        continue
                    else:
                        neighbor_node = neighbor_node.data
                        collision = check_collision(self.coords, neighbor_coords, ground_truth_map_info)
                        neighbor_matrix_x = center_index + (center_index - i)
                        neighbor_matrix_y = center_index + (center_index - j)
                        if not collision:
                            self.neighbor_matrix[i, j] = 1
                            self.neighbor_list.append(neighbor_coords)

                            neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                            neighbor_node.neighbor_list.append(self.coords)