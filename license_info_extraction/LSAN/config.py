import torch


TRAIN_FILE_PATH_TASK = './data/Task1.csv'
SAVE_PATH = './model/'
MAX_SEQ_LENGTH = 256

# hyper-parameters
NUM_EPOCHS = 100
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 0.0001
BETAS = (0.9, 0.99)
WEIGHT_DECAY = 0.00005

# model-parameters
embedding_size = 300
label_embedding_size = 768
lstm_hidden_dimension = 768
d_a = 200
num_of_class = 13
threshold = 0.5

# device configurations
use_cuda = False
if use_cuda and torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.cuda.set_device(2)
else:
    DEVICE = torch.device('cpu')
print(DEVICE)

