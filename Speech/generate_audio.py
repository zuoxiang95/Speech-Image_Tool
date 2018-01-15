# -*- coding: utf-8 -*-

from aip import AipSpeech
from os.path import join, isfile, isdir
from os import listdir, makedirs
from multiprocessing import Pool
import argparse as ap


# 定义常量
APP_ID = '9966136'
API_KEY = 'uEg7KD7y2wrDOia6TtAAI9pF'
SECRET_KEY = '1bbaeb8cd03df4e3997f30e4e64d2b34'

PERSON = 1
SPEED = 4


def synthesis(text):
    # 初始化AipSpeech对象
    aip_speech = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

    result = aip_speech.synthesis(text, 'zh', 1, {
        'per': PERSON,
        'spd': SPEED
    })

    return result


def write_mp3(mp3, output_dir):
    # 识别正确返回语音二进制，错误则返回dict 参照下面错误码
    if not isinstance(mp3, dict):
        with open(output_dir, 'wb') as f:
            f.write(mp3)


def download(text_path, output_path):
    with open(text_path, 'r') as f1:
        text = f1.readlines()
    j = 0
    for i in text:
        result = synthesis(i)

        output_mp3 = join(output_path, str(j) + '.mp3')
        write_mp3(result, output_mp3)
        output_txt = join(output_path, str(j) + '.txt')
        with open(output_txt, 'w') as f2:
            f2.write(i)
        print 'Downloaded %s' % i
        j = j + 1


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument(
        '--text_dir',
        help="The text input dir."
    )
    parser.add_argument(
        '--output_dir',
        help="The output dir."
    )
    args = vars(parser.parse_args())
    p = Pool(4)
    text_dir = args["text_dir"]
    output_dir = args["output_dir"]

    # 获取文件夹下所有文件名称
    text_list = [i for i in listdir(text_dir) if isfile(join(text_dir, i))]
    if not isdir(output_dir):
        makedirs(output_dir)

    for text_file in text_list:
        text_file_path = join(text_dir, text_file)
        output_path = join(output_dir, text_file.split('.')[0])
        if not isdir(output_path):
            makedirs(output_path)
        p.apply_async(download, args=(text_file_path, output_path))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('Process end')
