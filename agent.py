import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import copy

from rl_utils import *
from rl_parameter import *
from node_manager import NodeManager

class Agent:
    def __init__(self, policy_net, device='cpu', plot=False):
        self.device = device
        self.policy_net = policy_net
        self.plot = plot

        self.location = None
        self.map_info = None

        self.cell_size = CELL_SIZE
        self.node_resolution = NODE_RESOLUTION 
        self.updating_map_size = UPDATING_MAP_SIZE

        self.map_info = None
        self.updating_map_info = None

        self.frontier = set()

        self.node_manager = NodeManager(plot=self.plot)

        self.node_coords, self.utility, self.guidepost = None, None, None
        self.current_index, self.adjacent_matrix, self.neighbor_indices = None, None, None

    def update_map(self, map_info):
        self.map_info = map_info  

    def update_updating_map(self, location):
        self.updating_map_info = self.get_updating_map(location)

    def update_location(self, location):
        self.location = location
        node = self.node_manager.nodes_dict.find(location.tolist())
        if self.node_manager.nodes_dict.__len__() != 0:
            node.data.set_visited()

    def update_frontiers(self):
        self.frontier = get_frontier_in_map(self.updating_map_info)

    def get_updating_map(self, location):
        updating_map_origin_x = (location[0] - self.updating_map_size / 2)
        updating_map_origin_y = (location[1] - self.updating_map_size / 2)
        updating_map_top_x = updating_map_origin_x + self.updating_map_size
        updating_map_top_y = updating_map_origin_y + self.updating_map_size

        min_x = self.map_info.map_origin_x
        min_y = self.map_info.map_origin_y
        max_x = (self.map_info.map_origin_x + self.cell_size * (self.map_info.map.shape[1] - 1))
        max_y = (self.map_info.map_origin_y + self.cell_size * (self.map_info.map.shape[0] - 1))

        if updating_map_origin_x < min_x:
            updating_map_origin_x = min_x
        if updating_map_origin_y < min_y:
            updating_map_origin_y = min_y
        if updating_map_top_x > max_x:
            updating_map_top_x = max_x
        if updating_map_top_y > max_y:
            updating_map_top_y = max_y

        updating_map_origin_x = (updating_map_origin_x // self.cell_size + 1) * self.cell_size
        updating_map_origin_y = (updating_map_origin_y // self.cell_size + 1) * self.cell_size
        updating_map_top_x = (updating_map_top_x // self.cell_size) * self.cell_size
        updating_map_top_y = (updating_map_top_y // self.cell_size) * self.cell_size

        updating_map_origin_x = np.round(updating_map_origin_x, 1)
        updating_map_origin_y = np.round(updating_map_origin_y, 1)
        updating_map_top_x = np.round(updating_map_top_x, 1)
        updating_map_top_y = np.round(updating_map_top_y, 1)

        updating_map_origin = np.array([updating_map_origin_x, updating_map_origin_y])
        updating_map_origin_in_global_map = get_cell_position_from_coords(updating_map_origin, self.map_info)
        updating_map_top = np.array([updating_map_top_x, updating_map_top_y])
        updating_map_top_in_global_map = get_cell_position_from_coords(updating_map_top, self.map_info)

        updating_map = self.map_info.map[
            updating_map_origin_in_global_map[1]:updating_map_top_in_global_map[1]+1,
            updating_map_origin_in_global_map[0]:updating_map_top_in_global_map[0]+1
        ]
        updating_map_info = MapInfo(updating_map, updating_map_origin_x, updating_map_origin_y, self.cell_size)
        return updating_map_info

    def update_planning_state(self, map_info, location):
        self.update_map(map_info)
        self.update_location(location)
        self.update_updating_map(self.location)
        self.update_frontiers()
        self.node_manager.update_graph(self.location, self.frontier, self.updating_map_info)
        self.node_coords, self.utility, self.guidepost, self.adjacent_matrix, self.current_index, self.neighbor_indices = self.update_observation()

    def update_observation(self):
        all_node_coords = []
        for node in self.node_manager.nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = []

        shortest_path = self.compute_shortest_path()
        guidepost_flags = []
        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1]*1j

        for i, coords in enumerate(all_node_coords):
            node = self.node_manager.nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)

            if (coords[0], coords[1]) in shortest_path:
                guidepost_flags.append(1)
            else:
                guidepost_flags.append(0)
            for neighbor in node.neighbor_list:
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1]*1j)
                assert index is not None
                index = index[0][0]
                adjacent_matrix[i, index] = 0

        utility = np.array(utility)
        guidepost_flags = np.array(guidepost_flags)
        current_index = np.argwhere(node_coords_to_check == self.location[0] + self.location[1]*1j)[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)
        return all_node_coords, utility, guidepost_flags, adjacent_matrix, current_index, neighbor_indices

    def get_observation(self):
        node_coords = self.node_coords
        node_utility = self.utility.reshape(-1, 1)
        node_guidepost = self.guidepost.reshape(-1, 1)
        current_index = self.current_index
        edge_mask = self.adjacent_matrix
        current_edge = self.neighbor_indices
        n_node = node_coords.shape[0]

        current_node_coords = node_coords[self.current_index]
        node_coords = np.concatenate((node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                                      node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
                                      axis=-1) / UPDATING_MAP_SIZE / 2
        node_utility = node_utility / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        node_inputs = np.concatenate((node_coords, node_utility, node_guidepost), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
        assert node_coords.shape[0] < NODE_PADDING_SIZE, print(node_coords.shape[0], NODE_PADDING_SIZE)
        padding = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))
        node_inputs = padding(node_inputs)
        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)
        node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16).to(self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)
        current_index = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)
        edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)
        padding = torch.nn.ConstantPad2d((0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node), 1)
        edge_mask = padding(edge_mask)
        current_in_edge = np.argwhere(current_edge == self.current_index)[0][0]
        current_edge = torch.tensor(current_edge).unsqueeze(0)
        k_size = current_edge.size()[-1]
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
        current_edge = padding(current_edge)
        current_edge = current_edge.unsqueeze(-1)
        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)
        edge_padding_mask = padding(edge_padding_mask)
        return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask]

    def select_next_waypoint(self, observation):
        _, _, _, _, current_edge, _ = observation
        with torch.no_grad():
            logp = self.policy_net(*observation)
        action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)
        next_node_index = current_edge[0, action_index.item(), 0].item()
        next_position = self.node_coords[next_node_index]
        return next_position, action_index

    def plot_env(self):
        plt.switch_backend('agg')
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 2)
        nodes = get_cell_position_from_coords(self.node_coords, self.map_info)
        if len(self.frontier) > 0:
            frontiers = get_cell_position_from_coords(np.array(list(self.frontier)), self.map_info).reshape(-1, 2)
            plt.scatter(frontiers[:, 0], frontiers[:, 1], c='r', s=2)
        plt.imshow(self.map_info.map, cmap='gray')
        plt.axis('off')
        plt.scatter(nodes[:, 0], nodes[:, 1], c=self.utility, zorder=2)
        for node, utility in zip(nodes, self.utility):
            plt.text(node[0], node[1], str(utility), zorder=3)
        robot = get_cell_position_from_coords(self.location, self.map_info)
        plt.plot(robot[0], robot[1], 'mo', markersize=16, zorder=5)
        guidepost_mask = np.array(self.guidepost, dtype=bool)
        if guidepost_mask.any():
            guidepost_nodes = nodes[guidepost_mask]
            plt.scatter(guidepost_nodes[:, 0], guidepost_nodes[:, 1], c='green', s=50, marker='o', label='Guidepost', zorder=4)
            plt.legend()
        for coords in self.node_coords:
            node = self.node_manager.nodes_dict.find(coords.tolist()).data
            for neighbor_coords in node.neighbor_list[1:]:
                end = (np.array(neighbor_coords) - coords) / 2 + coords
                plt.plot((np.array([coords[0], end[0]]) - self.map_info.map_origin_x) / self.cell_size,
                         (np.array([coords[1], end[1]]) - self.map_info.map_origin_y) / self.cell_size,
                         'tan', zorder=1)

    def compute_shortest_path(self):
        dist, prev = self.node_manager.Dijkstra(self.location)
        best_dist = float('inf')
        best_node = None

        for node in self.node_manager.nodes_dict.__iter__():
            node_data = node.data
            if node_data.utility > 0:
                key = (node_data.coords[0], node_data.coords[1])
                if key == (self.location[0], self.location[1]):
                    continue
                if key in dist and dist[key] < best_dist:
                    best_dist = dist[key]
                    best_node = key
        if best_node is None:
            return []
        path, _ = self.node_manager.get_Dijkstra_path_and_dist(dist, prev, best_node)
        return path
