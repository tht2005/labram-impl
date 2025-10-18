import torch

def calculate_channel_phase(x: torch.Tensor):
    d = torch.fft.rfft(x, dim=-1)
    mag = d.abs()
    phase = d.angle()
    return (mag, phase)

