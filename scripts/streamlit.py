import os
import sys
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import librosa
import numpy as np
import streamlit as st
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.io import load_wav

st.title("Audio Classification")
audio = st.file_uploader("Upload a audio file", type=("mp3", "wav"))
SAMPLING_RATE = 22050
seq_len = 114688
device = "cpu"
yaml_dir = "/mnt/ebs/dev/AudioClassfication/outputs/kpf/kpf"


def crop_audio(audio):
    chucks = list(torch.split(audio, seq_len))
    if chucks[-1].size(0) < seq_len:
        chucks[-1] = F.pad(chucks[-1], (0, seq_len - chucks[-1].size(0)), "constant").data
    chucks = torch.stack(chucks)
    return chucks


def output_to_target(y_est):
    return y_est + 1


def infer(audios: torch.Tensor, net: nn.Module) -> np.ndarray:
    audios = audios.unsqueeze(1)
    audios.to(device)

    print("audios.mean(), audios.std()", audios.mean().item(), audios.std().item())
    pred = net(audios)
    # pred = torch.rand(audios.size(0), 8)
    y_est = torch.max(pred, 1)[1]
    print("y_est", y_est)
    target = output_to_target(y_est)
    return target.cpu()


def smoothing(outputs, alpha):
    return gaussian_filter1d(outputs, alpha)
    


def display_outputs(outputs, audio_length_seconds, is_half: bool):
    import matplotlib.pyplot as plt

    outputs = smoothing(outputs, 1.2)
    fig = plt.figure()
    step = seq_len / SAMPLING_RATE
    plt.xlim(0, audio_length_seconds)
    if is_half:
        plt.ylim(0, 5)
    else:
        plt.ylim(0, 9)
    steps = np.arange(step // 2, audio_length_seconds + step // 2, step)
    outputs = outputs.tolist()
    if len(steps) != len(outputs):
        outputs.append(outputs[-1])
    assert len(steps) == len(outputs), f"len(steps), {len(steps)} != len(outputs), {len(outputs)}"
    plt.scatter(
        steps,
        outputs,
        marker="_",
        s=SAMPLING_RATE // (len(outputs) - 1),
    )
    return fig


def parse_args(yaml_dir: str = yaml_dir, add_noise: bool = False):
    yaml_dir = Path(yaml_dir)
    with (yaml_dir / Path("args.yml")).open() as f:
        args = yaml.load(f, Loader=yaml.Loader)
    try:
        args = vars(args)
    except:
        if "net" in args.keys():
            del args["net"]
        args_orig = args
        args = {}
        for k, v in args_orig.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    args[kk] = vv
            else:
                args[k] = v
    args["f_res"] = yaml_dir
    args["add_noise"] = add_noise
    # with open(args['f_res'] / "args.yml", "w") as f:
    #     yaml.dump(args, f)
    return args


def build_model() -> nn.Module:
    args = parse_args()
    from modules.soundnet import SoundNetRaw as SoundNet

    ds_fac = np.prod(np.array(args["ds_factors"])) * 4
    net = SoundNet(
        nf=args["nf"],
        dim_feedforward=args["dim_feedforward"],
        clip_length=args["seq_len"] // ds_fac,
        embed_dim=args["emb_dim"],
        n_layers=args["n_layers"],
        nhead=args["n_head"],
        n_classes=args["n_classes"],
        factors=args["ds_factors"],
    )
    chkpnt = torch.load(args["f_res"] / "chkpnt.pt", map_location=torch.device(device))
    model = chkpnt["model_dict"]
    net.load_state_dict(model, strict=True)
    net.to(device)

    return net


if audio is not None:
    st.audio(audio)
    audio, sampling_rate = load_wav(audio)
    audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=SAMPLING_RATE)
    audio = torch.from_numpy(audio).type(torch.float32)
    audio = 0.95 * (audio / audio.__abs__().max()).float()

    if "net" not in st.session_state:
        st.session_state["net"] = build_model()
    audios = crop_audio(audio)
    outputs = infer(audios, st.session_state["net"]).numpy()
    fig = display_outputs(outputs, audio.shape[0] / SAMPLING_RATE, is_half=False)
    st.pyplot(fig)
    outputs_half = [(output + 1) // 2 for output in outputs]
    fig = display_outputs(outputs_half, audio.shape[0] / SAMPLING_RATE, is_half=True)
    st.pyplot(fig)
