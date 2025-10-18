import glob
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO
DATA_DIR = glob.glob('./data/OpenMIIR-RawEEG_v1/*.fif')
# DATA_DIR = ( './data/OpenMIIR-RawEEG_v1/P01-raw.fif', './data/OpenMIIR-RawEEG_v1/P04-raw.fif' )

BATCH_SIZE = 512
PATCH_t = 1024
PATCH_s = 384
PATCH_w = 64

assert PATCH_t % PATCH_w == 0
assert PATCH_w % 8 == 0

d_embd = PATCH_w
d_codebook = 8192

SPECTRUM_PRED_MODEL_WEIGHT_FILE = 'spectrum_pred_model_weights.pth'
