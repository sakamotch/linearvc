#!/usr/bin/env python

"""
Compute a male-to-female bias vector from two speaker directories.
"""

from pathlib import Path
import argparse
import torch

from linearvc import LinearVC, resolve_device

DEFAULT_SAMPLE_RATE = 16000


def check_argv():
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument(
        "male_wav_dir", type=Path, help="directory with male speaker speech"
    )
    parser.add_argument(
        "female_wav_dir", type=Path, help="directory with female speaker speech"
    )
    parser.add_argument("output_bias", type=Path, help="output bias vector filename")
    parser.add_argument(
        "--extension",
        choices=[".flac", ".wav"],
        help="audio file extension to read (default: '.wav')",
        default=".wav",
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


def list_audio_files(wav_dir: Path, extension: str) -> list[Path]:
    wavs = list(wav_dir.rglob("*" + extension))
    if not wavs:
        raise ValueError(f"No files ending with {extension} found in {wav_dir}")
    return wavs


def mean_features(linearvc_model: LinearVC, wavs: list[Path], vad: bool) -> torch.Tensor:
    total = None
    count = 0
    for wav_fn in sorted(wavs):
        feats = linearvc_model.get_features(wav_fn, vad=vad)
        if total is None:
            total = torch.zeros(feats.shape[1], device=feats.device)
        total += feats.sum(dim=0)
        count += feats.shape[0]
    if count == 0:
        raise ValueError("No frames found after feature extraction.")
    return total / count


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

    male_wavs = list_audio_files(args.male_wav_dir, args.extension)
    female_wavs = list_audio_files(args.female_wav_dir, args.extension)

    print("Computing mean male features...")
    male_mean = mean_features(linearvc_model, male_wavs, args.vad)
    print("Computing mean female features...")
    female_mean = mean_features(linearvc_model, female_wavs, args.vad)

    bias = female_mean - male_mean
    payload = {
        "bias": bias.detach().cpu(),
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "feature_dim": bias.numel(),
        "source": "male_to_female",
    }

    print("Writing:", args.output_bias)
    torch.save(payload, args.output_bias)


if __name__ == "__main__":
    args = check_argv()
    main(args)
