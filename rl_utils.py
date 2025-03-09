import numpy as np
import imageio
import os
from skimage.morphology import label
from PIL import Image
import torch
import torchvision.transforms as transforms
from parameter import *
from utils.tools import normalize

def get_cell_position_from_coords(coords, map_info, check_negative=True):
    single_cell = False
    if coords.flatten().shape[0] == 2:
        single_cell = True

    coords = coords.reshape(-1, 2)
    coords_x = coords[:, 0]
    coords_y = coords[:, 1]
    cell_x = ((coords_x - map_info.map_origin_x) / map_info.cell_size)
    cell_y = ((coords_y - map_info.map_origin_y) / map_info.cell_size)

    cell_position = np.around(np.stack((cell_x, cell_y), axis=-1)).astype(int)

    if check_negative:
        assert sum(cell_position.flatten() >= 0) == cell_position.flatten().shape[0], print(cell_position, coords, map_info.map_origin_x, map_info.map_origin_y)
    if single_cell:
        return cell_position[0]
    else:
        return cell_position


def get_coords_from_cell_position(cell_position, map_info):
    cell_position = cell_position.reshape(-1, 2)
    cell_x = cell_position[:, 0]
    cell_y = cell_position[:, 1]
    coords_x = cell_x * map_info.cell_size + map_info.map_origin_x
    coords_y = cell_y * map_info.cell_size + map_info.map_origin_y
    coords = np.stack((coords_x, coords_y), axis=-1)
    coords = np.around(coords, 1)
    if coords.shape[0] == 1:
        return coords[0]
    else:
        return coords


def get_free_area_coords(map_info):
    free_indices = np.where(map_info.map == FREE)
    free_cells = np.asarray([free_indices[1], free_indices[0]]).T
    free_coords = get_coords_from_cell_position(free_cells, map_info)
    return free_coords


def get_free_and_connected_map(location, map_info):
    # a binary map for free and connected areas
    free = (map_info.map == FREE).astype(float)
    labeled_free = label(free, connectivity=2)
    cell = get_cell_position_from_coords(location, map_info)
    label_number = labeled_free[cell[1], cell[0]]
    connected_free_map = (labeled_free == label_number)
    return connected_free_map


def get_updating_node_coords(location, updating_map_info, check_connectivity=True):
    x_min = updating_map_info.map_origin_x
    y_min = updating_map_info.map_origin_y
    x_max = updating_map_info.map_origin_x + (updating_map_info.map.shape[1] - 1) * CELL_SIZE
    y_max = updating_map_info.map_origin_y + (updating_map_info.map.shape[0] - 1) * CELL_SIZE

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

    free_connected_map = None

    if not check_connectivity:

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < updating_map_info.map.shape[0] and 0 <= cell[0] < updating_map_info.map.shape[1]
            if updating_map_info.map[cell[1], cell[0]] == FREE:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

    else:
        free_connected_map = get_free_and_connected_map(location, updating_map_info)
        free_connected_map = np.array(free_connected_map)

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < free_connected_map.shape[0] and 0 <= cell[0] < free_connected_map.shape[1]
            if free_connected_map[cell[1], cell[0]] == 1:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

    return nodes, free_connected_map


def get_frontier_in_map(map_info):
    x_len = map_info.map.shape[1]
    y_len = map_info.map.shape[0]
    unknown = (map_info.map == UNKNOWN) * 1
    unknown = np.lib.pad(unknown, ((1, 1), (1, 1)), 'constant', constant_values=0)
    unknown_neighbor = unknown[2:][:, 1:x_len + 1] + unknown[:y_len][:, 1:x_len + 1] + unknown[1:y_len + 1][:, 2:] \
                       + unknown[1:y_len + 1][:, :x_len] + unknown[:y_len][:, 2:] + unknown[2:][:, :x_len] + \
                       unknown[2:][:, 2:] + unknown[:y_len][:, :x_len]
    free_cell_indices = np.where(map_info.map.ravel(order='F') == FREE)[0]
    frontier_cell_1 = np.where(1 < unknown_neighbor.ravel(order='F'))[0]
    frontier_cell_2 = np.where(unknown_neighbor.ravel(order='F') < 8)[0]
    frontier_cell_indices = np.intersect1d(frontier_cell_1, frontier_cell_2)
    frontier_cell_indices = np.intersect1d(free_cell_indices, frontier_cell_indices)

    x = np.linspace(0, x_len - 1, x_len)
    y = np.linspace(0, y_len - 1, y_len)
    t1, t2 = np.meshgrid(x, y)
    cells = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    frontier_cell = cells[frontier_cell_indices]

    frontier_coords = get_coords_from_cell_position(frontier_cell, map_info).reshape(-1, 2)
    if frontier_cell.shape[0] > 0 and FRONTIER_CELL_SIZE != CELL_SIZE:
        frontier_coords = frontier_coords.reshape(-1 ,2)
        frontier_coords = frontier_down_sample(frontier_coords)
    else:
        frontier_coords = set(map(tuple, frontier_coords))
    return frontier_coords

def frontier_down_sample(data, voxel_size=FRONTIER_CELL_SIZE):
    voxel_indices = np.array(data / voxel_size, dtype=int).reshape(-1, 2)

    voxel_dict = {}
    for i, point in enumerate(data):
        voxel_index = tuple(voxel_indices[i])

        if voxel_index not in voxel_dict:
            voxel_dict[voxel_index] = point
        else:
            current_point = voxel_dict[voxel_index]
            if np.linalg.norm(point - np.array(voxel_index) * voxel_size) < np.linalg.norm(
                    current_point - np.array(voxel_index) * voxel_size):
                voxel_dict[voxel_index] = point

    downsampled_data = set(map(tuple, voxel_dict.values()))
    return downsampled_data


def check_collision(start, end, map_info):
    # Bresenham line algorithm checking
    assert start[0] >= map_info.map_origin_x
    assert start[1] >= map_info.map_origin_y
    assert end[0] >= map_info.map_origin_x
    assert end[1] >= map_info.map_origin_y
    assert start[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1]
    assert start[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0]
    assert end[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1]
    assert end[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0]
    collision = False

    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    map = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k == OCCUPIED:
            collision = True
            break
        if k == UNKNOWN:
            collision = True
            break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return collision


def make_gif(path, n, frame_files, rate):
    with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, rate), mode='I', duration=0.5) as writer:
        for frame in frame_files:
            image = imageio.imread(frame)
            writer.append_data(image)
    print('gif complete\n')

    # Remove files
    for filename in frame_files[:-1]:
        os.remove(filename)


class MapInfo:
    def __init__(self, map, map_origin_x, map_origin_y, cell_size, uncertainty=None):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
        self.cell_size = cell_size
        self.uncertainty = uncertainty


    def update_map_info(self, map, map_origin_x, map_origin_y):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y

def get_env_layout_mask(robot_belief, img_shape):
    layout = robot_belief.copy()
    mask = robot_belief.copy()
    mask[mask != 127] = 0
    mask[mask == 127] = 1

    layout = Image.fromarray(layout).convert('L')
    layout = transforms.Resize(img_shape[:-1])(layout)
    layout = transforms.CenterCrop(img_shape[:-1])(layout)
    layout = transforms.ToTensor()(layout)
    layout = normalize(layout)

    mask = Image.fromarray(mask*255).convert('L')
    mask = transforms.Resize(img_shape[:-1])(mask)
    mask = transforms.CenterCrop(img_shape[:-1])(mask)
    mask = np.array(mask)
    mask[mask > 0] = 255  # align with layout resize error
    mask = Image.fromarray(mask).convert('1')
    mask = transforms.ToTensor()(mask)[0].unsqueeze(dim=0)

    return layout, mask

def postprocess_layout(layout, robot_belief):
    img_shape = robot_belief.shape
    current_layout = robot_belief.copy()
    mask = robot_belief.copy()
    mask[mask != 127] = 0
    mask[mask == 127] = 1

    layout = layout.cpu().detach().squeeze()
    layout = transforms.ToPILImage()(layout)
    layout = transforms.Resize(img_shape)(layout)
    layout = transforms.CenterCrop(img_shape)(layout)
    layout = np.array(layout)

    layout[layout > 50] = 255
    layout = layout * mask + current_layout * (1 - mask)

    return layout


def make_gif(path, n, frame_files, rate):
    with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, rate), mode='I', duration=0.5) as writer:
        for frame in frame_files:
            image = imageio.imread(frame)
            writer.append_data(image)
    print('gif complete\n')

    # Remove files
    for filename in frame_files[:-1]:
        os.remove(filename)


class MapInfo:
    def __init__(self, map, map_origin_x, map_origin_y, cell_size, uncertainty=None):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
        self.cell_size = cell_size
        self.uncertainty = uncertainty


    def update_map_info(self, map, map_origin_x, map_origin_y):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y


import numpy as np

import numpy as np

def discretize_uncertainty_map(uncertainty_map, q1=33, q2=66):

    discretized_map = np.zeros_like(uncertainty_map, dtype=np.uint8)

    flat_map = uncertainty_map.flatten().astype(float)
    valid_values = flat_map[flat_map > 0]
    
    if len(valid_values) == 0:
        return discretized_map
    
    low_threshold = np.percentile(valid_values, q1)   
    high_threshold = np.percentile(valid_values, q2)  

    valid_mask = (uncertainty_map > 0)
    
    low_mask = (valid_mask & (uncertainty_map < low_threshold))
    mid_mask = (valid_mask & (uncertainty_map >= low_threshold) & (uncertainty_map < high_threshold))
    high_mask = (valid_mask & (uncertainty_map >= high_threshold))
    
    discretized_map[low_mask] = 1
    discretized_map[mid_mask] = 2
    discretized_map[high_mask] = 3

    return discretized_map

def postprocess_layout_for_uncertainty(std_pred, robot_belief):

    img_shape = robot_belief.shape
    # mask: unknown=127 ->1, known=0
    mask = robot_belief.copy()
    mask[mask != 127] = 0
    mask[mask == 127] = 1

    if isinstance(std_pred, torch.Tensor):
        std_pred = std_pred.cpu().detach().squeeze()  # shape: [H,W]
    std_img = transforms.ToPILImage()(std_pred)
    std_img = transforms.Resize(img_shape)(std_img)
    std_img = transforms.CenterCrop(img_shape)(std_img)
    std_map = np.array(std_img, dtype=np.float32)

    std_map = std_map * mask

    return std_map

def postprocess_inpainting_and_uncertainty(mean_pred, std_pred, robot_belief):

    inpainted_layout = postprocess_layout(mean_pred, robot_belief)  

    std_map = postprocess_layout_for_uncertainty(std_pred, robot_belief)

    uncertainty_map = np.zeros_like(std_map, dtype=np.float32)
    kept_mask = (inpainted_layout == 255)  
    uncertainty_map[kept_mask] = std_map[kept_mask]

    return inpainted_layout, uncertainty_map
