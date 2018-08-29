# -*- coding: utf-8 -*-
"""
    colorthief
    ~~~~~~~~~~

    Grabbing the color palette from an image.

    :copyright: (c) 2015 by Shipeng Feng.
    :license: BSD, see LICENSE for more details.
"""
__version__ = '0.2.1'

from ctypes import *
import ctypes

import numpy as np

import matplotlib.pyplot as plt

import time

#from queue import PriorityQueue as PQueue


class cached_property(object):
    """Decorator that creates converts a method with a single
    self argument into a property cached on the instance.
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, type):
        res = instance.__dict__[self.func.__name__] = self.func(instance)
        return res


class ColorThief(object):
    """Color thief main class."""
    def __init__(self, im):
        """Create one color thief for one image.

        :param file: A filename (string) or a file object. The file object
                     must implement `read()`, `seek()`, and `tell()` methods,
                     and be opened in binary mode.
        """
        self.image = im
        #matplotlib.image.pil_to_array(pilImage)
        #Load a PIL image and return it as a numpy array.

    def get_color(self, quality=10):
        """Get the dominant color.

        :param quality: quality settings, 1 is the highest quality, the bigger
                        the number, the faster a color will be returned but
                        the greater the likelihood that it will not be the
                        visually most dominant color
        :return tuple: (r, g, b)
        """
        color_count = 5                  ####原先的程序中设置为5，为何？

        ##优化前0.032  优化之后是0.001+
        width, height, channel = self.image.shape
        pixels = np.reshape(self.image, (width*height, channel))

        filt = np.arange(0, width * height, quality)
        pixels = pixels[filt, :]

        ##0.0005+
        valid_pixels = pixels[np.nonzero(((pixels[:, 0] <= 250) | (pixels[:, 1] <= 250) | (pixels[:, 2] <= 250)))[0], :]

        valid_pixels = valid_pixels.astype(np.int32)   # np.int32 -> np.int64： 会导致后面某些部分运算时间变短了

        # Send array to quantize function which clusters values
        # using median cut algorithm
        dominant_color = MMCQ.quantize(valid_pixels, color_count)
        return dominant_color


class MMCQ(object):
    """Basic Python port of the MMCQ (modified median cut quantization)
    algorithm from the Leptonica library (http://www.leptonica.com/).
    """

    SIGBITS = 7               #将颜色由RGB各8位压缩至5位
    RSHIFT = 8 - SIGBITS
    MAX_ITERATION = 1000
    FRACT_BY_POPULATIONS = 0.75

    @staticmethod
    def get_histo(pixels):
        """histo (1-d array, giving the number of pixels in each quantized
        region of color space)
        """
        pixels = pixels >> MMCQ.RSHIFT                #pixels矩阵中每个元素都右移

        sz = pow(2, MMCQ.SIGBITS)
        histo = np.zeros((sz, sz, sz), dtype='int32')

        calcHist = ctypes.cdll.LoadLibrary("../calcHist.so")
        dataPtr = pixels.ctypes.data_as(ctypes.c_char_p)
        histoPtr = histo.ctypes.data_as(ctypes.c_char_p)
        calcHist.calcHist(dataPtr, pixels.shape[0], histoPtr, sz)

        return histo

    @staticmethod
    def vbox_from_pixels(pixels, histo):
        rmin = 1000000
        rmax = 0
        gmin = 1000000
        gmax = 0
        bmin = 1000000
        bmax = 0
        pixels = pixels >> MMCQ.RSHIFT
        rmin = min(pixels[:, 0].min(), rmin)
        rmax = max(pixels[:, 0].max(), rmax)
        gmin = min(pixels[:, 1].min(), gmin)
        gmax = max(pixels[:, 1].max(), gmax)
        bmin = min(pixels[:, 2].min(), bmin)
        bmax = max(pixels[:, 2].max(), bmax)
        return VBox(rmin, rmax, gmin, gmax, bmin, bmax, histo)

    @staticmethod
    def median_cut_apply(histo, vbox):
        if not vbox.count:
            return (None, None)

        rw = vbox.r2 - vbox.r1 + 1
        gw = vbox.g2 - vbox.g1 + 1
        bw = vbox.b2 - vbox.b1 + 1
        maxw = max([rw, gw, bw])
        # only one pixel, no split
        if vbox.count == 1:
            return (vbox.copy, None)
        # Find the partial sum arrays along the selected axis.
        total = 0
        partialsum = {}
        lookaheadsum = {}
        do_cut_color = None
        indexes = []
        if maxw == rw:
            do_cut_color = 'r'
            for i in range(vbox.r1, vbox.r2+1):
                total += histo[i, vbox.g1:vbox.g2+1, vbox.b1:vbox.b2+1].sum()
                partialsum[i] = total

        elif maxw == gw:
            do_cut_color = 'g'
            for j in range(vbox.g1, vbox.g2+1):
                total += histo[vbox.r1:vbox.r2+1, j, vbox.b1:vbox.b2+1].sum()
                partialsum[j] = total

        else:  # maxw == bw
            do_cut_color = 'b'
            for k in range(vbox.b1, vbox.b2+1):
                total += histo[vbox.r1:vbox.r2+1, vbox.g1:vbox.g2+1, k].sum()
                partialsum[k] = total

        for i, d in partialsum.items():
            lookaheadsum[i] = total - d

        # determine the cut planes
        dim1 = do_cut_color + '1'
        dim2 = do_cut_color + '2'
        dim1_val = getattr(vbox, dim1)
        dim2_val = getattr(vbox, dim2)
        for i in range(dim1_val, dim2_val+1):    ##循环次数小
            if partialsum[i] > (total / 2):
                vbox1 = vbox.copy
                vbox2 = vbox.copy
                left = i - dim1_val
                right = dim2_val - i
                if left <= right:
                    d2 = min([dim2_val - 1, int(i + right / 2)])
                else:
                    d2 = max([dim1_val, int(i - 1 - left / 2)])
                # avoid 0-count boxes
                while not partialsum.get(d2, False):    ##循环次数小
                    d2 += 1
                count2 = lookaheadsum.get(d2)
                while not count2 and partialsum.get(d2-1, False):
                    d2 -= 1
                    count2 = lookaheadsum.get(d2)
                # set dimensions
                setattr(vbox1, dim2, d2)
                setattr(vbox2, dim1, getattr(vbox1, dim2) + 1)
                return (vbox1, vbox2)
        return (None, None)

    @staticmethod
    def quantize(pixels, max_color):
        """Quantize.

        :param pixels: a list of pixel in the form (r, g, b)
        :param max_color: max number of colors
        """
        if not pixels.shape[0]:
            raise Exception('Empty pixels when quantize.')
        if max_color < 2 or max_color > 256:
            raise Exception('Wrong number of max colors when quantize.')

        ########0.0004
        histo = MMCQ.get_histo(pixels)        ##c优化

        # check that we aren't below maxcolors already
        if len(histo) <= max_color:
            # generate the new colors from the histo and return
            pass

        #########0.0002
        # get the beginning vbox from the colors
        vbox = MMCQ.vbox_from_pixels(pixels, histo)

        pq = PQueue('x.count')                  #以count排序
        pq.push(vbox)

        # inner function to do the iteration
        def iter_(lh, target):
            n_color = 1
            n_iter = 0
            while n_iter < MMCQ.MAX_ITERATION:

                vbox = lh.pop()                   ##这部分耗时：0.013 0.012 0.010等   #pop()远比push耗时，是因为pop()包含排序操作   改成for循环后，时间效率没有太大的变化

                if not vbox.count:                ##如果像素数为0  just put it back   ##这部分耗时： e-06
                    lh.push(vbox)
                    n_iter += 1
                    return
                    #continue

                # do the cut
                ##这部分耗时：0.009 0.007 0.006等
                vbox1, vbox2 = MMCQ.median_cut_apply(histo, vbox)

                ##接下来这部分不耗时：8.82148742676e-06
                if not vbox1:
                    raise Exception("vbox1 not defined; shouldn't happen!")
                lh.push(vbox1)
                if vbox2:  # vbox2 can be null
                    lh.push(vbox2)
                    n_color += 1

                if n_color >= target:
                    return
                if n_iter > MMCQ.MAX_ITERATION:
                    return
                n_iter += 1

        ###########0.001
        # first set of colors, sorted by population
        iter_(pq, MMCQ.FRACT_BY_POPULATIONS * max_color)

        ###这里不新建pq2，继续使用pq,不过修改排序规则
        pq.set_sort_key('x.count*x.volume')

        ###########0.0002
        # next set - generate the median cuts using the (npix * vol) sorting.
        iter_(pq, max_color - pq.size())

        #########0.0007
        dominant_color = pq.pop().avg

        return dominant_color

class VBox(object):
    """3d color space box"""
    def __init__(self, r1, r2, g1, g2, b1, b2, histo):
        self.r1 = r1
        self.r2 = r2
        self.g1 = g1
        self.g2 = g2
        self.b1 = b1
        self.b2 = b2
        self.histo = histo

    @cached_property
    def volume(self):
        sub_r = self.r2 - self.r1
        sub_g = self.g2 - self.g1
        sub_b = self.b2 - self.b1
        return (sub_r + 1) * (sub_g + 1) * (sub_b + 1)

    @property
    def copy(self):
        return VBox(self.r1, self.r2, self.g1, self.g2,
                    self.b1, self.b2, self.histo)

    @cached_property
    def count(self):
        npix = int(self.histo[self.r1:self.r2+1, self.g1:self.g2+1, self.b1:self.b2+1].sum())        #耗时e-05  如果使用 sum(histo_mat[lower_ind:upper_ind + 1, 1]) 耗时非常高
        return npix

    @cached_property
    def avg(self):
        mult = 1 << (8 - MMCQ.SIGBITS)
        r_sum = 0
        g_sum = 0
        b_sum = 0
        for i in range(self.r1, self.r2 + 1):
            r_sum += self.histo[i, self.g1:self.g2 + 1, self.b1:self.b2 + 1].sum() * (i+0.5) * mult
        for j in range(self.g1, self.g2 + 1):
            g_sum += self.histo[self.r1:self.r2 + 1, j, self.b1:self.b2 + 1].sum() * (j+0.5) * mult
        for k in range(self.b1, self.b2 + 1):
            b_sum += self.histo[self.r1:self.r2 + 1, self.g1:self.g2 + 1, k].sum() * (k+0.5) * mult

        if self.count:
            r_avg = int(r_sum / self.count)
            g_avg = int(g_sum / self.count)
            b_avg = int(b_sum / self.count)
        else:
            r_avg = int(mult * (self.r1 + self.r2 + 1) / 2)
            g_avg = int(mult * (self.g1 + self.g2 + 1) / 2)
            b_avg = int(mult * (self.b1 + self.b2 + 1) / 2)

        return r_avg, g_avg, b_avg

class PQueue(object):
    """Simple priority queue."""
    def __init__(self, sort_key):
        self.sort_key = sort_key
        self.contents = []        ##用list实现PQ???

    def set_sort_key(self, sort_key):
        self.sort_key = sort_key

    def push(self, o):
        self.contents.append(o)

    def pop(self):
        ind = -1
        max = -1
        if self.sort_key == 'x.count':      ##不排序，直接返回最大count对应的vbox
           for i in range(self.size()):
               cnt = self.contents[i].count
               if cnt > max:
                   max = cnt
                   ind = i
        elif self.sort_key == 'x.count*x.volume':
            for i in range(self.size()):
                cnt = self.contents[i].count * self.contents[i].volume
                if cnt > max:
                    max = cnt
                    ind = i
        ret = self.contents[ind]
        del self.contents[ind]
        return ret

    def size(self):
        return len(self.contents)
