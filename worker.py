from time import time
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from collections import deque

from env import Env
from agent import Agent
from rl_utils import *
from rl_model import PolicyNet
from ground_truth_node_manager import GroundTruthNodeManager, MapInfo

from rl_parameter import *
from model.networks import Generator, prune_generator, remove_prune_reparam


IMG_SHAPE = (256, 256, 1)
TEST_METHOD = 'rl-generative'

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class Worker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image)
        self.robot = Agent(policy_net, self.device, self.save_image)

        self.ground_truth_node_manager = GroundTruthNodeManager(
            self.robot.node_manager, 
            self.env.ground_truth_info,
            device=self.device, 
            plot=self.save_image
        )

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(27):
            self.episode_buffer.append([])

        gan_config = {
            'netG': {
                'input_dim': 1,       
                'past_channels': 5,   
                'ngf': 16
            }
        }
        self.generator = Generator(
            config=gan_config, 
            use_cuda=(self.device.type == 'cuda'), 
            device_ids=[self.device.index]
        ).to(self.device)
        gan_checkpoint_path = "checkpoints/map_small/hole_map_inpainting/gen_best.pt"
        self.generator.load_state_dict(torch.load(gan_checkpoint_path, map_location=self.device))

        prune_amount = 0.1
        self.generator = prune_generator(self.generator, amount=prune_amount)
        self.generator = remove_prune_reparam(self.generator)

        self.generator.eval()
        print("Worker {}: GAN Generator loaded, pruned and set to eval mode.".format(self.meta_agent_id))
        
        self.predicted_map_info = None

        self.past_layouts = deque(maxlen=5)

    def run_episode(self):
        done = False
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        observation = self.robot.get_observation()
        ground_truth_observation = self.ground_truth_node_manager.get_ground_truth_observation(self.env.robot_location)

        if self.save_image:
            self.robot.plot_env()
            self.ground_truth_node_manager.plot_ground_truth_env(self.env.robot_location)
            self.env.plot_env(0)

        for i in range(MAX_EPISODE_STEP):
            current_layout, mask = get_env_layout_mask(self.env.robot_belief, IMG_SHAPE)
            masked_layout = current_layout * (1. - mask)
            masked_layout = masked_layout.unsqueeze(0).to(self.device)  # [1,1,H,W]
            mask = mask.unsqueeze(0).to(self.device)                     # [1,1,H,W]

            self.past_layouts.append(masked_layout)

            num_samples = 3
            predictions = []
            self.generator.train() 
            with torch.no_grad():
                for _ in range(num_samples):
                    if len(self.past_layouts) < 5:
                        past_tensor = None
                    else:
                        past_tensor = torch.cat(list(self.past_layouts), dim=1)
                    _, x2_sample, _ = self.generator(masked_layout, mask, past=past_tensor, dropout=True)
                    x2_inpaint_sample = x2_sample * mask + masked_layout * (1. - mask)
                    predictions.append(x2_inpaint_sample.unsqueeze(0))
            predictions = torch.cat(predictions, dim=0)
            mean_pred = predictions.mean(dim=0)
            std_pred = predictions.std(dim=0)
            self.generator.eval()  

            inpainted_layout, uncertainty_map =  postprocess_inpainting_and_uncertainty(
                                                mean_pred, 
                                                std_pred, 
                                                self.env.robot_belief)
         
            uncertainty_map = discretize_uncertainty_map(uncertainty_map, q1=33, q2=66)

            new_uncertainty = np.sum(uncertainty_map)

            self.env.last_uncertainty = new_uncertainty

            self.predicted_map_info = MapInfo(
                inpainted_layout,
                self.env.belief_origin_x,
                self.env.belief_origin_y,
                self.env.cell_size,
                uncertainty=uncertainty_map
            )
            self.ground_truth_node_manager_predicted = GroundTruthNodeManager(
                self.robot.node_manager,
                self.predicted_map_info,
                device=self.device,
                plot=self.save_image
            )

            observation = self.robot.get_observation()
            state = self.ground_truth_node_manager_predicted.get_ground_truth_observation(self.env.robot_location)
            ground_truth_observation = self.ground_truth_node_manager.get_ground_truth_observation(self.env.robot_location)                        

            self.save_observation(state, ground_truth_observation)
            #self.save_observation(state, state)

            next_location, action_index = self.robot.select_next_waypoint(state)
            self.save_action(action_index)

            node = self.robot.node_manager.nodes_dict.find((self.robot.location[0], self.robot.location[1]))
            check = np.array(node.data.neighbor_list)
            assert next_location[0] + next_location[1] * 1j in check[:, 0] + check[:, 1] * 1j, \
                print(next_location, self.robot.location, node.data.neighbor_list)
            assert next_location[0] != self.robot.location[0] or next_location[1] != self.robot.location[1]

            reward = self.env.step(next_location)

            self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
            if self.robot.utility.sum() == 0:
                done = True
                reward += 20
            self.save_reward_done(reward, done)

            observation = self.robot.get_observation()
            ground_truth_observation = self.ground_truth_node_manager.get_ground_truth_observation(
                self.env.robot_location)
            state = self.ground_truth_node_manager_predicted.get_ground_truth_observation(self.env.robot_location)
            self.save_next_observations(state, ground_truth_observation)
            #self.save_next_observations(state, state)

            if self.save_image:
                self.robot.plot_env()
                self.ground_truth_node_manager_predicted.plot_ground_truth_env(self.env.robot_location)
                self.env.plot_env(i+1)

            if done:
                break

        self.perf_metrics['travel_dist'] = self.env.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def save_observation(self, state, ground_truth_observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = state
        self.episode_buffer[0] += node_inputs
        self.episode_buffer[1] += node_padding_mask.bool()
        self.episode_buffer[2] += edge_mask.bool()
        self.episode_buffer[3] += current_index
        self.episode_buffer[4] += current_edge
        self.episode_buffer[5] += edge_padding_mask.bool()
        
        critic_node_inputs, critic_node_padding_mask, critic_edge_mask, critic_current_index, critic_current_edge, critic_edge_padding_mask = ground_truth_observation
        self.episode_buffer[15] += critic_node_inputs
        self.episode_buffer[16] += critic_node_padding_mask.bool()
        self.episode_buffer[17] += critic_edge_mask.bool()
        self.episode_buffer[18] += critic_current_index
        self.episode_buffer[19] += critic_current_edge
        self.episode_buffer[20] += critic_edge_padding_mask.bool()

        assert torch.all(current_edge == critic_current_edge), \
            print(current_edge, critic_current_edge, current_index, critic_current_index)
        assert torch.all(node_inputs[0, current_index.item(), :2] == critic_node_inputs[0, critic_current_index.item(), :2]), \
            print(node_inputs[0, current_index.item()], critic_node_inputs[0, critic_current_index.item()])
        assert torch.all(torch.gather(node_inputs, 1, current_edge.repeat(1, 1, 2)) ==
                         torch.gather(critic_node_inputs, 1, critic_current_edge.repeat(1, 1, 2)))

    def save_action(self, action_index):
        self.episode_buffer[6] += action_index.reshape(1, 1, 1)

    def save_reward_done(self, reward, done):
        self.episode_buffer[7] += torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device)
        self.episode_buffer[8] += torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)

    def save_next_observations(self, state, ground_truth_observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = state
        self.episode_buffer[9] += node_inputs
        self.episode_buffer[10] += node_padding_mask.bool()
        self.episode_buffer[11] += edge_mask.bool()
        self.episode_buffer[12] += current_index
        self.episode_buffer[13] += current_edge
        self.episode_buffer[14] += edge_padding_mask.bool()

        critic_node_inputs, critic_node_padding_mask, critic_edge_mask, critic_current_index, critic_current_edge, critic_edge_padding_mask = ground_truth_observation
        self.episode_buffer[21] += critic_node_inputs
        self.episode_buffer[22] += critic_node_padding_mask.bool()
        self.episode_buffer[23] += critic_edge_mask.bool()
        self.episode_buffer[24] += critic_current_index
        self.episode_buffer[25] += critic_current_edge
        self.episode_buffer[26] += critic_edge_padding_mask.bool()

        assert torch.all(current_edge == critic_current_edge), \
            print(current_edge, critic_current_edge, current_index, critic_current_index)
        assert torch.all(node_inputs[0, current_index.item(), :2] == critic_node_inputs[0, critic_current_index.item(), :2]), \
            print(node_inputs[0, current_index.item()], critic_node_inputs[0, critic_current_index.item()])
        assert torch.all(torch.gather(node_inputs, 1, current_edge.repeat(1, 1, 2)) ==
                         torch.gather(critic_node_inputs, 1, critic_current_edge.repeat(1, 1, 2)))

if __name__ == "__main__":
    torch.manual_seed(4777)
    np.random.seed(4777)
    model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
    # checkpoint = torch.load(model_path + '/checkpoint.pth', map_location='cpu')
    # model.load_state_dict(checkpoint['policy_model'])
    worker = Worker(0, model, 77, save_image=False)
    worker.run_episode()
