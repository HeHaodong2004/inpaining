# saving path
FOLDER_NAME = 'ariadne1_coverage'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'

# save training data
SUMMARY_WINDOW = 32
LOAD_MODEL = False
SAVE_IMG_GAP = 400

# map and planning resolution
CELL_SIZE = 0.4  # meter
NODE_RESOLUTION = 4.0  # meter
FRONTIER_CELL_SIZE = 2 * CELL_SIZE

# map representation
FREE = 255
OCCUPIED = 1
UNKNOWN = 127

# sensor and utility range
SENSOR_RANGE = 16  # meter
UTILITY_RANGE = 0.8 * SENSOR_RANGE
MIN_UTILITY = 5

# updating map range w.r.t the robot
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION

# training parameters
MAX_EPISODE_STEP = 128
REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 1500
BATCH_SIZE = 128
LR = 1e-5
GAMMA = 1
NUM_META_AGENT = 1

# network parameters
NODE_INPUT_DIM = 5
EMBEDDING_DIM = 128

# Graph parameters
K_SIZE = 25  # the number of neighboring nodes
NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value

# GPU usage
USE_GPU = False  # do you want to collect training data using GPUs
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 1

