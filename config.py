DATA_PATH = "data"
FEATURE_PATH = "data/feature"
MODE = "multi"

# Features
SAMPLE_RATE = 22050
FFT_SIZE = 1024
WIN_SIZE = 1024
HOP_SIZE = 512
NUM_MELS = 40
FEATURE_LENGTH = 1024
LABEL_NAME = "Label_abs_16"
# 'arousal','valence','Label_kMean_4','Label_kMean_16','Label_kMean_64','Label_abs_4','Label_abs_16','Label_abs_64'

# Training
DEVICE = "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-2
STOPPING_RATE = 1e-5
WEIGHT_DECAY = 1e-6
MOMENTUM = 0.9
FACTOR = 0.2
PATIENCE = 3
MAX_LEN = 50
EMBD_DIM = 100
LSTM_LATENT_DIM = 40