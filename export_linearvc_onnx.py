#!/usr/bin/env python

"""
Export WavLM and HiFiGAN to ONNX using a real input waveform.
"""

from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F

from linearvc import (
    resolve_device,
    DEFAULT_SAMPLE_RATE,
    VAD_TRIGGER_LEVEL,
)


class WavLMFeatures(nn.Module):
    def __init__(self, wavlm, output_layer: int):
        super().__init__()
        self.wavlm = wavlm
        self.output_layer = output_layer

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        feats, _ = self.wavlm.extract_features(wav, output_layer=self.output_layer)
        return feats


def check_argv():
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("input_wav", type=Path, help="input speech filename")
    parser.add_argument("wavlm_onnx", type=Path, help="output WavLM ONNX filename")
    parser.add_argument(
        "hifigan_onnx", type=Path, help="output HiFiGAN ONNX filename"
    )
    parser.add_argument(
        "--output-layer",
        type=int,
        default=6,
        help="WavLM output layer to export (default: 6)",
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        help="apply voice activity detection to trim leading silence",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="compute device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--hub-dir",
        type=Path,
        help=(
            "path to a local clone of https://github.com/bshall/knn-vc to"
            " load torch.hub modules without network access"
        ),
    )
    return parser.parse_args()


def load_wav_for_export(path: Path, device: str, vad: bool) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    wav = wav.to(device)
    if sr != DEFAULT_SAMPLE_RATE:
        wav = F.resample(wav, orig_freq=sr, new_freq=DEFAULT_SAMPLE_RATE)
    if vad:
        wav_cpu = wav.cpu()
        trimmed = F.vad(
            wav_cpu,
            sample_rate=DEFAULT_SAMPLE_RATE,
            trigger_level=VAD_TRIGGER_LEVEL,
        )
        if trimmed.numel() == 0:
            trimmed = wav_cpu
        wav = trimmed.to(device)
    return wav


def main(args):
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    if args.hub_dir is not None:
        if not args.hub_dir.exists():
            raise FileNotFoundError(args.hub_dir)
        repo_or_dir = str(args.hub_dir)
        hub_source = "local"
    else:
        repo_or_dir = "bshall/knn-vc"
        hub_source = "github"

    wavlm = torch.hub.load(
        repo_or_dir,
        "wavlm_large",
        trust_repo=True,
        progress=True,
        device=device,
        source=hub_source,
    ).eval()
    hifigan, _ = torch.hub.load(
        repo_or_dir,
        "hifigan_wavlm",
        trust_repo=True,
        prematched=True,
        progress=True,
        device=device,
        source=hub_source,
    )
    hifigan = hifigan.eval().to(device)

    wav = load_wav_for_export(args.input_wav, device=device, vad=args.vad)

    wavlm_wrapper = WavLMFeatures(wavlm, args.output_layer).eval()
    with torch.no_grad():
        feats = wavlm_wrapper(wav)
    # Ensure a normal tensor for tracing (not inference-mode).
    feats = feats.detach().clone()

    print("Exporting WavLM:", args.wavlm_onnx)
    torch.onnx.export(
        wavlm_wrapper,
        wav,
        args.wavlm_onnx,
        input_names=["wav"],
        output_names=["features"],
        opset_version=args.opset,
        dynamo=False,
    )

    print("Exporting HiFiGAN:", args.hifigan_onnx)
    torch.onnx.export(
        hifigan,
        feats,
        args.hifigan_onnx,
        input_names=["features"],
        output_names=["wav"],
        opset_version=args.opset,
        dynamo=False,
    )


if __name__ == "__main__":
    args = check_argv()
    main(args)
