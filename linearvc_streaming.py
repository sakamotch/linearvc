#!/usr/bin/env python

"""
Streaming voice conversion using LinearVC from microphone to speaker.
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import torch

from linearvc import LinearVC, resolve_device, DEFAULT_SAMPLE_RATE


def check_argv():
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument(
        "source_wav_dir", type=Path, help="directory with source speaker speech"
    )
    parser.add_argument(
        "target_wav_dir", type=Path, help="directory with target speaker speech"
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=1000,
        help="streaming chunk size in milliseconds (default: 1000)",
    )
    parser.add_argument(
        "--extension",
        choices=[".flac", ".wav"],
        help="source and target audio file extension (default: '.wav')",
        default=".wav",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="whether source and target utterances are parallel",
    )
    parser.add_argument(
        "--lasso", type=float, help="lasso is applied with this alpha value"
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
    parser.add_argument(
        "--input-device",
        help="input device index or name (default: system default)",
    )
    parser.add_argument(
        "--output-device",
        help="output device index or name (default: system default)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="list audio devices and exit",
    )
    return parser.parse_args()


def list_audio_files(wav_dir: Path, extension: str) -> list[Path]:
    wavs = list(wav_dir.rglob("*" + extension))
    if not wavs:
        raise ValueError(f"No files ending with {extension} found in {wav_dir}")
    return wavs


def resolve_device_index(device_arg):
    if device_arg is None:
        return None
    try:
        return int(device_arg)
    except ValueError:
        return device_arg


def main(args):
    try:
        import sounddevice as sd
    except Exception as exc:
        print("sounddevice is required for streaming I/O:", exc, file=sys.stderr)
        sys.exit(1)

    if args.list_devices:
        print(sd.query_devices())
        return

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

    source_wavs = list_audio_files(args.source_wav_dir, args.extension)
    target_wavs = list_audio_files(args.target_wav_dir, args.extension)

    W = linearvc_model.get_projmat(
        source_wavs,
        target_wavs,
        parallel=args.parallel,
        lasso=args.lasso,
        vad=False,
    )

    chunk_samples = int(DEFAULT_SAMPLE_RATE * args.chunk_ms / 1000)
    if chunk_samples <= 0:
        raise ValueError("chunk-ms must be > 0")

    input_device = resolve_device_index(args.input_device)
    output_device = resolve_device_index(args.output_device)

    stream_in = sd.InputStream(
        samplerate=DEFAULT_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=chunk_samples,
        device=input_device,
    )
    stream_out = sd.OutputStream(
        samplerate=DEFAULT_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=chunk_samples,
        device=output_device,
    )

    print("Starting streaming conversion... Press Ctrl+C to stop.")
    with stream_in, stream_out, torch.inference_mode():
        while True:
            indata, _ = stream_in.read(chunk_samples)
            wav = torch.from_numpy(indata.T).to(device)
            features, _ = wavlm.extract_features(wav, output_layer=6)
            input_features = features.squeeze(0)

            output_wav = linearvc_model.project_and_vocode(input_features, W)
            out_np = output_wav.numpy().astype(np.float32)

            if out_np.shape[0] < chunk_samples:
                pad = chunk_samples - out_np.shape[0]
                out_np = np.pad(out_np, (0, pad))
            else:
                out_np = out_np[:chunk_samples]

            stream_out.write(out_np[:, None])


if __name__ == "__main__":
    args = check_argv()
    main(args)
