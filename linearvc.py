#!/usr/bin/env python

"""
Perform voice conversion with linear regression.

Author: Herman Kamper
Date: 2024
"""

from pathlib import Path
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F

from utils import fast_cosine_dist

n_frames_max = 8192  # maximum no. of matched frames in linear regression
k_top = 1
VAD_TRIGGER_LEVEL = 7.0
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LinearVC(nn.Module):
    def __init__(
        self,
        wavlm,
        hifigan,
        device: str = DEFAULT_DEVICE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        super().__init__()
        self.device = device
        self.sr = sample_rate
        self.wavlm = wavlm.eval().to(self.device)
        self.hifigan = hifigan.eval().to(self.device)

    @torch.inference_mode()
    def get_features(self, wav_fn, vad=False):
        """
        Return features of `wav_fn` as a tensor with shape (n_frames, dim).

        VAD can optionally be applied to remove leading silence.
        """

        wav, sr = torchaudio.load(wav_fn)
        wav = wav.to(self.device)

        if sr != self.sr:
            wav = F.resample(wav, orig_freq=sr, new_freq=self.sr)

        # Trim silence at beginning (if specified)
        if vad:
            wav_cpu = wav.cpu()
            trimmed = F.vad(
                wav_cpu,
                sample_rate=self.sr,
                trigger_level=VAD_TRIGGER_LEVEL,
            )
            if trimmed.numel() == 0:
                trimmed = wav_cpu
            wav = trimmed.to(self.device)

        features, _ = self.wavlm.extract_features(wav, output_layer=6)
        features = features.squeeze()

        return features

    @torch.inference_mode()
    def get_projmat(
        self, source_wavs, target_wavs, parallel=False, lasso=None, vad=False
    ):
        if parallel and lasso is None:
            lasso = 0.3

        if not source_wavs:
            raise ValueError("No source waveforms were provided.")
        if not target_wavs:
            raise ValueError("No target waveforms were provided.")

        if not parallel:
            # Source features
            source_features = []
            print("Source features:")
            for wav_fn in tqdm(sorted(source_wavs), leave=True):
                source_features.append(self.get_features(wav_fn, vad))
            source_features = torch.vstack(source_features)[:n_frames_max, :]

            # Target features
            target_features = []
            print("Target features:")
            for wav_fn in tqdm(sorted(target_wavs), leave=True):
                target_features.append(self.get_features(wav_fn, vad))
            target_features = torch.vstack(target_features)[:n_frames_max, :]

            # Matching
            dists = fast_cosine_dist(
                source_features, target_features, device=self.device
            )
            best = dists.topk(k=k_top, largest=False, dim=-1)
            linear_target = target_features[best.indices].mean(dim=1)
        else:
            # Audio with the same name: parallel utterance pairs
            source_map = {wav_fn.name: wav_fn for wav_fn in source_wavs}
            target_map = {wav_fn.name: wav_fn for wav_fn in target_wavs}
            common_files = sorted(source_map.keys() & target_map.keys())
            if not common_files:
                raise ValueError(
                    "Parallel mode requested but no matching filenames were found."
                )
            source_target_wav_pairs = [
                (source_map[name], target_map[name]) for name in common_files
            ]

            # Inputs and outputs for linear regression
            combined_source_feats = []
            combined_linear_target = []
            for source_wav_fn, target_wav_fn in tqdm(source_target_wav_pairs):
                # Features
                source_features = self.get_features(source_wav_fn, vad)
                target_features = self.get_features(target_wav_fn, vad)

                # Matching
                dists = fast_cosine_dist(
                    source_features, target_features, device=self.device
                )
                best = dists.topk(k=k_top, largest=False, dim=-1)
                linear_target = target_features[best.indices].mean(dim=1)

                combined_source_feats.append(source_features)
                combined_linear_target.append(linear_target)

            source_features = torch.vstack(combined_source_feats)
            linear_target = torch.vstack(combined_linear_target)

        # Projection matrix
        source_np = source_features.detach().cpu().numpy()
        linear_target_np = linear_target.detach().cpu().numpy()

        if lasso is None:
            from numpy import linalg

            W, _, _, _ = linalg.lstsq(
                source_np,
                linear_target_np,
                rcond=None,
            )
        else:
            import celer

            print(f"Lasso with alpha: {lasso:.2f}")

            linear = celer.Lasso(alpha=lasso, fit_intercept=False).fit(
                source_np,
                linear_target_np,
            )
            W = linear.coef_.T

        W = torch.from_numpy(W).float().to(self.device)
        return W

    @torch.inference_mode()
    def project_and_vocode(self, input_features, W):
        """Return the waveform samples."""
        source_to_target_feats = input_features[None] @ W
        wav_hat = self.hifigan(source_to_target_feats).squeeze(0)
        return wav_hat.cpu().squeeze().cpu()


def check_argv():
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument(
        "source_wav_dir", type=Path, help="directory with source speaker speech"
    )
    parser.add_argument(
        "target_wav_dir", type=Path, help="directory with target speaker speech"
    )
    parser.add_argument("input_wav", type=Path, help="input speech filename")
    parser.add_argument("output_wav", type=Path, help="output speech filename")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="whether source and target utterances are parallel,"
        " in which case the filenames in the two directories should match",
    )
    parser.add_argument(
        "--lasso", type=float, help="lasso is applied with this alpha value"
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        help="voice activatiy detecion is applied to start of utterance",
    )
    parser.add_argument(
        "--extension",
        choices=[".flac", ".wav"],
        help="source and target audio file extension (default: '.wav')",
        default=".wav",
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


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch reports that CUDA is unavailable.")
    return device_arg


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

    # Load the WavLM feature extractor and HiFiGAN vocoder
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

    linearvc_model = LinearVC(wavlm, hifigan, device)

    # Lists of source and target audio files
    print("Reading from:", args.source_wav_dir)
    source_wavs = list(args.source_wav_dir.rglob("*" + args.extension))
    if not source_wavs:
        raise ValueError(
            f"No source files ending with {args.extension} found in {args.source_wav_dir}"
        )
    print("Reading from:", args.target_wav_dir)
    target_wavs = list(args.target_wav_dir.rglob("*" + args.extension))
    if not target_wavs:
        raise ValueError(
            f"No target files ending with {args.extension} found in {args.target_wav_dir}"
        )

    # Features for the source input utterance
    print("Reading:", args.input_wav)
    input_features = linearvc_model.get_features(args.input_wav)

    # The voice conversion projection matrix
    W = linearvc_model.get_projmat(
        source_wavs,
        target_wavs,
        parallel=args.parallel,
        lasso=args.lasso,
        vad=args.vad,
    )

    # Project the input and vocode
    output_wav = linearvc_model.project_and_vocode(input_features, W)

    print("Writing:", args.output_wav)
    torchaudio.save(args.output_wav, output_wav[None], linearvc_model.sr)


if __name__ == "__main__":
    args = check_argv()
    main(args)
