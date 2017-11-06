# -*- coding: utf-8 -*-

from pydub import AudioSegment
import wave
import io
import argparse as ap
from os import listdir, makedirs
from os.path import isdir, isfile, join


def mp3_translate_wav(mp3_path, wav_path):
    """
        将mp3格式的音频文件转化为wav格式的音频
    :param mp3_path: mp3文件的详细路径
    :param wav_path: 输出的wav文件的路径，精确到输出文件名
    :return: None
    """
    with open(mp3_path, 'rb') as f1:
        data = f1.read()
    audio_data = io.BytesIO(data)
    sound = AudioSegment.from_file(audio_data, format='mp3')
    raw_data = sound._data
    raw_length = len(raw_data)
    f = wave.open(wav_path, 'wb')
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)
    f.setnframes(raw_length)
    f.writeframes(raw_data)
    f.close()


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument(
        '--mp3_dir',
        help="The mp3 audio path."
    )
    parser.add_argument(
        '--output_path',
        help="The audio output path."
    )
    args = vars(parser.parse_args())
    mp3_dir = args["mp3_dir"]
    output_path = args["output_path"]
    mp3_translate_wav(mp3_dir, output_path)
