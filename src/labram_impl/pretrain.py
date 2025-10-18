from .config import BATCH_SIZE, PATCH_s, PATCH_t, PATCH_w
from .config import d_embd, d_codebook
from .config import DATA_DIR

import torch
import torch.optim as optim
import numpy as np

from tqdm import trange, tqdm

import mne

from .model import *

def get_raw_eeg_data():
    eeg_raw_list = []
    for file in DATA_DIR:
        raw = mne.io.read_raw_fif(file)
        eeg_channels = mne.pick_types(raw.info, eeg=True)
        eeg_data = raw.get_data(picks=eeg_channels)
        sfreq = raw.info['sfreq']
        # change the sampling rate to ?
        #print(eeg_channels)
        #print(eeg_data)
        assert len(eeg_channels) == len(eeg_data)
        for i in range(1, len(eeg_data)):
            assert len(eeg_data[0]) == len(eeg_data[i])
        eeg_raw_list.append((eeg_channels, eeg_data))
    return eeg_raw_list

def patch(eeg_raw_list):
    patch_list = []
    for (channels, data) in eeg_raw_list:
        T = len(data[0])
        tmp2 = []
        for i in range(0, T - PATCH_t, PATCH_s):
            tmp = []
            for j in range(i, i + PATCH_t - PATCH_w, PATCH_w):
                tmp.extend(data[c][j:j+PATCH_w] for c in range(len(channels)))
            tmp2.append(tmp)
        patch_list.append((channels, np.array(tmp2)))
    return patch_list

def vq_neural_spectrum_prediction_train(eeg_patch_list):
    BATCH_SIZE = 1024
    PEAK_LR = 5e-5
    MIN_LR = 1e-5
    adambeta = (0.9, 0.99)
    weight_decay = 1e-4
    tot_epochs = 2
    # tot_epochs = 100
    wu_epochs = 10
    data_stride = 200

    model = NeuralTokenizerTrainer(BATCH_SIZE, d_embd, d_codebook)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=PEAK_LR,
        betas=adambeta,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=tot_epochs, 
        eta_min=MIN_LR
    )

    print(f"Training on {device}")
    for epoch in trange(tot_epochs, desc="Epochs"):
        model.train()
        running_loss = 0.0

        for (channels, data) in eeg_patch_list:
            np.random.shuffle(data)
            batch_iter = tqdm(
                # TODO
                range(0, len(eeg_patch_list), BATCH_SIZE),
                # range(0, 3 * BATCH_SIZE, BATCH_SIZE),
                desc=f"Epoch {epoch+1}/{tot_epochs}",
                leave=False,
                ncols=100
            )
            for i in batch_iter:
                batch_np = data[i:i+BATCH_SIZE]
                batch = torch.from_numpy(batch_np).float().to(device, non_blocking=True)

                loss = model(batch).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                avg_loss = running_loss / ((i // BATCH_SIZE) + 1)

                batch_iter.set_postfix(loss=f"{avg_loss:.6f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            scheduler.step()

if __name__ == "__main__":
    eeg_raw_list = get_raw_eeg_data()
    eeg_patch_list = patch(eeg_raw_list)
    model = vq_neural_spectrum_prediction_train(eeg_patch_list)
    # torch.save(model.state_dict(), "spectrum_pred_model_weights.pth")

