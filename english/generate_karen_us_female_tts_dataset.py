#!/usr/bin/env python3

import argparse
from pathlib import Path
import sherpa_onnx

from dataclasses import dataclass
import soundfile as sf


@dataclass
class Item:
    idx: str
    text: str


def read_transcript(filename, start, stop):
    with open(filename, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= stop:
                break

            line = line.strip()
            fields = line.split("|")
            if len(fields) != 2:
                print(f"skip {line}")
            text = fields[1].replace(".", ",")
            #  text = fields.replace('?', ',')
            #  text = fields.replace('!', ',')

            yield Item(idx=fields[0], text=text)


def process_transcript(tts, start, transcript, dest_data_dir):
    base_dir = Path(dest_data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    num_speakers = tts.num_speakers

    k = start
    num_per_folder = 100

    for item in transcript:
        k += 1
        if k % num_speakers == 0:
            print(k)

        for spk_id in range(num_speakers):
            folder = k // num_per_folder
            audio_id = f"{spk_id}-{item.idx}.flac"
            d = base_dir / f"{spk_id}/{folder}"
            d.mkdir(parents=True, exist_ok=True)
            f = d / audio_id
            if f.is_file():
                continue
            audio = tts.generate(item.text, sid=spk_id, speed=1.0)
            sf.write(f"{f}", audio.samples, samplerate=audio.sample_rate)


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dest-data-dir",
        type=str,
        required=True,
        help="Directory to save the generated audio",
    )

    parser.add_argument(
        "--transcript",
        type=str,
        required=True,
        help="Path to the transcript",
    )

    parser.add_argument(
        "--tts-model-dir",
        type=str,
        required=True,
        help="Path to tts model directory",
    )

    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Start line in the transcript to process (inclusive)",
    )

    parser.add_argument(
        "--stop",
        type=int,
        required=True,
        help="Stop line in the transcript to process (exclusive)",
    )

    return parser.parse_args()


def create_tts(model_dir):
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=f"{model_dir}/model.onnx",
                voices=f"{model_dir}/voices.bin",
                tokens=f"{model_dir}/tokens.txt",
                data_dir=f"{model_dir}/espeak-ng-data",
                dict_dir=f"{model_dir}/dict",
                lexicon=f"{model_dir}/lexicon-us-en.txt,{model_dir}/lexicon-zh.txt",
            ),
            provider="cpu",
            debug=False,
            num_threads=2,
        ),
        max_num_sentences=1,
    )
    if not tts_config.validate():
        raise ValueError("Please check your config")
    return sherpa_onnx.OfflineTts(tts_config)


def main():
    args = get_args()
    print(vars(args))
    tts = create_tts(args.tts_model_dir)
    transcript = read_transcript(args.transcript, start=args.start, stop=args.stop)
    process_transcript(tts, args.start, transcript, args.dest_data_dir)


if __name__ == "__main__":
    main()
