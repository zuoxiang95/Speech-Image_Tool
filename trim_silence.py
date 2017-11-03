# -*- coding: utf-8 -*-

import librosa
import numpy as np
import argparse as ap


def trim_silence(audio_path, sample_rate, threshold, frame_length=2048):
    """
    Removes silence at the beginning and end of a sample.

    :param audio_path: the path of your audio.
    :param sample_rate: the sample rate of audio.
    :param threshold: anything quieter than this will be considered silence.
    :param frame_length:
    :return: the audio which removes the silence at beginning and end
    """
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    if audio.size < frame_length:
        frame_length = audio.size
    energy = librosa.feature.rmse(audio, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument(
        '--audio_path',
        help="The audio path."
    )
    parser.add_argument(
        '--output_path',
        help="The audio output path."
    )
    args = vars(parser.parse_args())
    audio_path = args["audio_path"]
    output_path = args["output_path"]
    trim_silence_audio = trim_silence(audio_path, sample_rate=16000, threshold=40)
    write_wav(trim_silence_audio, sample_rate=16000, filename=output_path)
