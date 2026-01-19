#!/usr/bin/env python

"""
Apply a bias vector to convert speech between male and female style.
"""

from pathlib import Path
import argparse
import torch
import torchaudio

from linearvc import LinearVC, resolve_device

DEFAULT_SAMPLE_RATE = 16000


def check_argv():
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("bias_vector", type=Path, help="bias vector file")
    parser.add_argument("input_wav", type=Path, help="input speech filename")
    parser.add_argument("output_wav", type=Path, help="output speech filename")
    parser.add_argument(
        "--direction",
        choices=["m2f", "f2m"],
        required=True,
        help="conversion direction: m2f (male->female) or f2m (female->male)",
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        help="apply voice activity detection to trim leading silence",
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


def load_bias_vector(path: Path, device: str) -> torch.Tensor:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "bias" in payload:
        bias = payload["bias"]
    else:
        bias = payload
    if not torch.is_tensor(bias):
        bias = torch.from_numpy(bias)
    return bias.float().to(device)


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
    )
    hifigan, _ = torch.hub.load(
        repo_or_dir,
        "hifigan_wavlm",
        trust_repo=True,
        prematched=True,
        progress=True,
        device=device,
        source=hub_source,
    )

    linearvc_model = LinearVC(wavlm, hifigan, device, sample_rate=DEFAULT_SAMPLE_RATE)

    bias = load_bias_vector(args.bias_vector, device=device)
    if args.direction == "f2m":
        bias = -bias

    print("Reading:", args.input_wav)
    input_features = linearvc_model.get_features(args.input_wav, vad=args.vad)
    if input_features.shape[1] != bias.numel():
        raise ValueError(
            f"Bias dim {bias.numel()} does not match feature dim {input_features.shape[1]}"
        )

    source_to_target_feats = input_features + bias
    output_wav = linearvc_model.hifigan(source_to_target_feats[None]).squeeze(0)

    print("Writing:", args.output_wav)
    torchaudio.save(args.output_wav, output_wav.cpu()[None], linearvc_model.sr)


if __name__ == "__main__":
    args = check_argv()
    main(args)
