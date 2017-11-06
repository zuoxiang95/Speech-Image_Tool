# -*- coding: utf-8 -*-

from pypinyin import Style, lazy_pinyin
""" 
  注意，这里需要对 pypinyin.style._constants._INITIALS_NOT_STRICT 进行修改，
  添加自定义声母
  _INITIALS_NOT_STRICT = _INITIALS + ['y', 'w', 'ee','aa','oo','vv']
"""
from pypinyin.style._utils import get_initials, get_finals
from os import makedirs, listdir
from os.path import isfile, join, isdir
import argparse as ap

TRANSLATE_DICT = {
        'ai': 'aaai',
        'an': 'aaan',
        'ao': 'aaao',
        'ang': 'aaang',
        'e': 'eee',
        'er': 'eeer',
        'ou': 'ooou',
        'o': 'ooo',
        'si': 'six',
        'zi': 'zix',
        'ci': 'cix',
        'ri': 'rix',
        'shi': 'shix',
        'zhi': 'zhix',
        'chi': 'chix',
        'yu': 'vvv',
        'yue': 'vvve',
        'yun': 'vvvn',
        'yuan': 'vvvan',
        'xu': 'xv',
        'xun': 'xvn',
        'xue': 'xve',
        'xuan': 'xvan',
        'ju': 'jv',
        'jue': 'jve',
        'jun': 'jvn',
        'juan': 'jvan',
        'qu': 'qv',
        'que': 'qve',
        'qun': 'qvn',
        'quan': 'qvan',
        'lue': 'lve',
        'nue': 'nve'
    }


def translate_pinyin(sentence):
    """
        将中文语句（只能包含中文字符和中文标点符号）转化为模型需要的拼音
    :param sentence: 输入的中文语句
    :return: 返回转化后的拼音数据
    """
    # 对中文语句进行编码转化，转化为utf-8编码格式
    sentence = sentence.decode(encoding='utf-8')
    # 对中文语句进行转化
    pinyin_list = lazy_pinyin(sentence, style=Style.TONE3)
    result = []
    # 对转化的拼音的格式进行修改
    for pinyin in pinyin_list:
        # 判断当前拼音是否为标点符号
        if pinyin in [u'\uff0c', u'\u3002', u'\uff1f', u'\uff01', u'\u3001']:
            result.append(pinyin)
            continue
        '''对拼音进行标准化，更换声母和部分韵母的表达方式，以及对一些轻音的添加声调为第5声'''
        # 对没有声调的轻音，将其转化为第5声
        if pinyin[-1] not in ['1', '2', '3', '4']:
            pinyin = pinyin + '5'
        # 按照 TRANSLATE_DICT 中，对部分拼音进行更换声母或者韵母
        if pinyin[:-1] in TRANSLATE_DICT.keys():
            pinyin = TRANSLATE_DICT[pinyin[:-1]] + pinyin[-1]

        # 获取声母
        shengmu = get_initials(pinyin, strict=False)
        # 获取韵母
        yunmu = get_finals(pinyin, strict=False)
        # 将单个拼音按照“{声母 韵母}”的格式输出
        result.append('{' + shengmu + ' ' + yunmu + '}')
    # 返回中文语句转化成拼音的字符串
    return ' '.join(result)


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument(
        '--text_path',
        help="The chinese text path."
    )
    parser.add_argument(
        '--output_path',
        help="The pinyin output path."
    )
    args = vars(parser.parse_args())
    text_path = args["text_path"]
    output_path = args["output_path"]
    # 打开文本文档
    with open(text_path, 'r') as f1:
        tmp = f1.readlines()

    with open(output_path, 'w') as f2:
        for text in tmp:
            # 将汉字转化为拼音
            pinyin_tmp = translate_pinyin(text.strip())
            # 将转化后的拼音写入到文本文件中
            f2.write(pinyin_tmp.encode(encoding='gb2312'))
