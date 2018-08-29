# -*- coding: utf-8 -*-

import time

import matplotlib.image as mpimg
from numpy import *

import re
import os

from colorthief import ColorThief


CSS3_NAMES_TO_HEX = {
    u'aliceblue': u'#f0f8ff',
    u'antiquewhite': u'#faebd7',
    u'aqua': u'#00ffff',
    u'aquamarine': u'#7fffd4',
    u'azure': u'#f0ffff',
    u'beige': u'#f5f5dc',
    u'bisque': u'#ffe4c4',
    u'black': u'#000000',
    u'blanchedalmond': u'#ffebcd',
    u'blue': u'#0000ff',
    u'blueviolet': u'#8a2be2',
    u'brown': u'#a52a2a',
    u'burlywood': u'#deb887',
    u'cadetblue': u'#5f9ea0',
    u'chartreuse': u'#7fff00',
    u'chocolate': u'#d2691e',
    u'coral': u'#ff7f50',
    u'cornflowerblue': u'#6495ed',
    u'cornsilk': u'#fff8dc',
    u'crimson': u'#dc143c',
    u'cyan': u'#00ffff',
    u'darkblue': u'#00008b',
    u'darkcyan': u'#008b8b',
    u'darkgoldenrod': u'#b8860b',
    u'darkgray': u'#a9a9a9',
    u'darkgreen': u'#006400',
    u'darkkhaki': u'#bdb76b',
    u'darkmagenta': u'#8b008b',
    u'darkolivegreen': u'#556b2f',
    u'darkorange': u'#ff8c00',
    u'darkorchid': u'#9932cc',
    u'darkred': u'#8b0000',
    u'darksalmon': u'#e9967a',
    u'darkseagreen': u'#8fbc8f',
    u'darkslateblue': u'#483d8b',
    u'darkslategray': u'#2f4f4f',
    u'darkturquoise': u'#00ced1',
    u'darkviolet': u'#9400d3',
    u'deeppink': u'#ff1493',
    u'deepskyblue': u'#00bfff',
    u'dimgray': u'#696969',
    u'dodgerblue': u'#1e90ff',
    u'firebrick': u'#b22222',
    u'floralwhite': u'#fffaf0',
    u'forestgreen': u'#228b22',
    u'fuchsia': u'#ff00ff',
    u'gainsboro': u'#dcdcdc',
    u'ghostwhite': u'#f8f8ff',
    u'gold': u'#ffd700',
    u'goldenrod': u'#daa520',
    u'gray': u'#808080',
    u'green': u'#008000',
    u'greenyellow': u'#adff2f',
    u'honeydew': u'#f0fff0',
    u'hotpink': u'#ff69b4',
    u'indianred': u'#cd5c5c',
    u'indigo': u'#4b0082',
    u'ivory': u'#fffff0',
    u'khaki': u'#f0e68c',
    u'lavender': u'#e6e6fa',
    u'lavenderblush': u'#fff0f5',
    u'lawngreen': u'#7cfc00',
    u'lemonchiffon': u'#fffacd',
    u'lightblue': u'#add8e6',
    u'lightcoral': u'#f08080',
    u'lightcyan': u'#e0ffff',
    u'lightgoldenrodyellow': u'#fafad2',
    u'lightgray': u'#d3d3d3',
    u'lightgreen': u'#90ee90',
    u'lightpink': u'#ffb6c1',
    u'lightsalmon': u'#ffa07a',
    u'lightseagreen': u'#20b2aa',
    u'lightskyblue': u'#87cefa',
    u'lightslategray': u'#778899',
    u'lightsteelblue': u'#b0c4de',
    u'lightyellow': u'#ffffe0',
    u'lime': u'#00ff00',
    u'limegreen': u'#32cd32',
    u'linen': u'#faf0e6',
    u'magenta': u'#ff00ff',
    u'maroon': u'#800000',
    u'mediumaquamarine': u'#66cdaa',
    u'mediumblue': u'#0000cd',
    u'mediumorchid': u'#ba55d3',
    u'mediumpurple': u'#9370db',
    u'mediumseagreen': u'#3cb371',
    u'mediumslateblue': u'#7b68ee',
    u'mediumspringgreen': u'#00fa9a',
    u'mediumturquoise': u'#48d1cc',
    u'mediumvioletred': u'#c71585',
    u'midnightblue': u'#191970',
    u'mintcream': u'#f5fffa',
    u'mistyrose': u'#ffe4e1',
    u'moccasin': u'#ffe4b5',
    u'navajowhite': u'#ffdead',
    u'navy': u'#000080',
    u'oldlace': u'#fdf5e6',
    u'olive': u'#808000',
    u'olivedrab': u'#6b8e23',
    u'orange': u'#ffa500',
    u'orangered': u'#ff4500',
    u'orchid': u'#da70d6',
    u'palegoldenrod': u'#eee8aa',
    u'palegreen': u'#98fb98',
    u'paleturquoise': u'#afeeee',
    u'palevioletred': u'#db7093',
    u'papayawhip': u'#ffefd5',
    u'peachpuff': u'#ffdab9',
    u'peru': u'#cd853f',
    u'pink': u'#ffc0cb',
    u'plum': u'#dda0dd',
    u'powderblue': u'#b0e0e6',
    u'purple': u'#800080',
    u'red': u'#ff0000',
    u'rosybrown': u'#bc8f8f',
    u'royalblue': u'#4169e1',
    u'saddlebrown': u'#8b4513',
    u'salmon': u'#fa8072',
    u'sandybrown': u'#f4a460',
    u'seagreen': u'#2e8b57',
    u'seashell': u'#fff5ee',
    u'sienna': u'#a0522d',
    u'silver': u'#c0c0c0',
    u'skyblue': u'#87ceeb',
    u'slateblue': u'#6a5acd',
    u'slategray': u'#708090',
    u'snow': u'#fffafa',
    u'springgreen': u'#00ff7f',
    u'steelblue': u'#4682b4',
    u'tan': u'#d2b48c',
    u'teal': u'#008080',
    u'thistle': u'#d8bfd8',
    u'tomato': u'#ff6347',
    u'turquoise': u'#40e0d0',
    u'violet': u'#ee82ee',
    u'wheat': u'#f5deb3',
    u'white': u'#ffffff',
    u'whitesmoke': u'#f5f5f5',
    u'yellow': u'#ffff00',
    u'yellowgreen': u'#9acd32',
}

"""
将141种颜色归类到下面的12种，直接计算距离不准确(非线性的)
"""
color_rgb = {
    "Blue": ["navy", "darkblue", "mediumblue", "blue", "midnightblue", "royalblue", "lightblue", "paleturquoise",
             "lightsteelblue", "powderblue"],
    "Yellow": ["palegoldenrod", "khaki", "lightgoldenrodyellow", "gold", "yellow", "lightyellow"],
    "Red": ["maroon", "darkred", "firebrick", "mediumvioletred", "indianred", "crimson", "lightcoral", "salmon",
            "red", "orangered", "tomato"],
    "Green": ["darkgreen", "green", "mediumspringgreen", "lime", "springgreen", "forestgreen", "seagreen",
              "darkslategray", "limegreen", "mediumseagreen",
              "lawngreen", "chartreuse", "darkseagreen", "lightgreen", "palegreen", "greenyellow", "yellowgreen"],
    "Purple": ["darkslateblue", "indigo", "slateblue", "mediumslateblue", "purple", "blueviolet", "darkmagenta",
               "mediumpurple",
               "darkviolet", "darkorchid", "mediumorchid", "violet", "fuchsia", "magenta"],
    "White": ["silver", "gainsboro", "aliceblue", "honeydew", "azure", "whitesmoke", "mintcream", "ghostwhite",
              "antiquewhite", "floralwhite", "snow", "ivory", "white"],
    "Black": ["black"],
    "Gray": ["dimgray", "slategray", "lightslategray", "gray", "darkgray", "lightgray", "lightcyan", "lavender"],
    "Brown": ["darkolivegreen", "olivedrab", "olive", "saddlebrown", "sienna", "brown", "rosybrown",
              "darkkhaki", "burlywood", "sandybrown"],
    "Pink": ["thistle", "orchid", "palevioletred", "plum", "deeppink", "hotpink", "lightpink", "pink",
             "mistyrose", "lavenderblush", "seashell"],
    "Teal": ["teal", "darkcyan", "deepskyblue", "darkturquoise", "aqua", "cyan", "dodgerblue", "lightseagreen",
             "turquoise", "steelblue", "mediumturquoise", "cadetblue", "cornflowerblue", "mediumaquamarine",
             "aquamarine", "skyblue", "lightskyblue"],
    "Orange": ["darkgoldenrod", "peru", "chocolate", "tan", "goldenrod", "darksalmon", "wheat", "beige",
               "linen", "oldlace", "coral", "darkorange", "lightsalmon", "orange", "peachpuff", "navajowhite",
               "moccasin", "bisque", "blanchedalmond", "papayawhip", "cornsilk", "lemonchiffon"],
}

HEX_COLOR_RE = re.compile(r'^#([a-fA-F0-9]{3}|[a-fA-F0-9]{6})$')


def normalize_hex(hex_value):
    """
    Normalize a hexadecimal color value to 6 digits, lowercase.

    """
    match = HEX_COLOR_RE.match(hex_value)
    if match is None:
        raise ValueError(
            u"'{}' is not a valid hexadecimal color value.".format(hex_value)
        )
    hex_digits = match.group(1)
    if len(hex_digits) == 3:
        hex_digits = u''.join(2 * s for s in hex_digits)
    return u'#{}'.format(hex_digits.lower())


def hex_to_rgb(hex_value):
    """
    Convert a hexadecimal color value to a 3-tuple of integers
    suitable for use in an ``rgb()`` triplet specifying that color.

    """
    hex_value = normalize_hex(hex_value)
    hex_value = int(hex_value[1:], 16)
    return (
        hex_value >> 16,
        hex_value >> 8 & 0xff,
        hex_value & 0xff
    )


def css_color_to_base_color(css_color_str):
    for name, value in color_rgb.items():
        if css_color_str in value:
            return name

    return ""


def closest_colour(requested_colour):
    """
    用空间距离把rgb归类到颜色值上，但是颜色分布不均衡，不是线性的
    所以，先归类到140种，然后再通过人工定义的颜色映射归类到12种颜色上
    :param requested_colour:
    :return:
    """
    min_colours = {}
    for name, value in CSS3_NAMES_TO_HEX.items():
        r_c, g_c, b_c = hex_to_rgb(value)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    closest_colour_name = min_colours[min(min_colours.keys())]
    return css_color_to_base_color(closest_colour_name)

'''
m = np.array([[1,11], [2,12], [3,13], [4,14]])
ind = np.array([[4]])
y = m[:,0]==ind
z = np.nonzero(y)
x = sum(m[z[1], 1])
'''

'''
dic = {1:11,  3:33, 2:22, 4:44}
keys=dic.keys()
vals=dic.values()
t = zip(keys,vals)
tmp = np.array(t)
'''
#L=[key, val for key,val in zip(keys,vals) ]


#a=mat([[1,1,1],[1,1,1],[1,1,1]])

#c = (a[:,0] << (2 * 1)) + (a[:,1] << 1) + a[:,2]

#ptsInCurrCluster = a[np.nonzero((a[:, 3] > 4) & ((a[:, 0] < 7) | (a[:, 1] < 7) | (a[:, 2] < 7) ) )[0], :]

#nn = ptsInCurrCluster[:, 0:3]

'''
im = mpimg.imread('photo1.jpg')  ##需要20ms

#print mpcol.is_color_like(im)

#im = mpcol.to_rgba_array(im)

#im = Image.open('photo1.jpg')
start = time.time()

color_thief = ColorThief(im)
dominant_color = color_thief.get_color(quality=10)
print(dominant_color)

end = time.time()
print end-start

'''

dir = '/home/dr/color-thief-py/val2017'
fileList = os.listdir(dir)
L = []
T = 0

for s in fileList:

    newDir = os.path.join(dir, s)
    im = mpimg.imread(newDir)

    start = time.time()

    color_thief = ColorThief(im)
    dominant_color = color_thief.get_color(quality=1)
    color_type = closest_colour(dominant_color)

    end = time.time()

    T += (end - start)

    L.append(color_type)

print T

f = open('/home/dr/color-thief-py/Matrix-C-colortype-quality1-bit7.txt', 'w+')

for item in L:
    f.write(item)
    f.write('\n')

f.close()
