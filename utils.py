import numpy as np
from scipy.ndimage import distance_transform_edt, distance_transform_cdt, distance_transform_bf
import PIL.Image as Image
from PIL import ImageFilter
from typing import Tuple
import cv2 as cv

MAGIC = 255

def _get_distance_map(img_array: np.array, thresh: int = 10, pow: float = 0.5) -> np.array:
    # Relative distance to nearest black pixel
    black_mask = np.all(img_array[:, :, :3] <= [thresh, thresh, thresh], axis=-1) & (img_array[:, :, 3] >= MAGIC - thresh)
    dis_map = distance_transform_edt(~black_mask)
    return np.pow(dis_map / np.max(dis_map), pow)

def gradient_map(img: Image, thresh: int = 10, pow: float = 0.5) -> Image:
    # Gradient map for 3D shading assistance
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    img_array = np.array(img)
    output_image = Image.new('RGBA', img_array.shape[0:2])
    dis_map = _get_distance_map(img_array, thresh=thresh, pow=pow)
    pixels = output_image.load()

    for i in range(output_image.size[0]):
        for j in range(output_image.size[1]):
            pixels[i,j] = (int(MAGIC*dis_map[i,j]),
                            int(MAGIC*dis_map[i,j]),
                            int(MAGIC*dis_map[i,j]),
                            int(img_array[i,j,3]*dis_map[i,j]))

    return output_image

def outline_simplify(img: Image) -> Image:
    # Colours the inside of the image white, while outlining the edges in black
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    imgf = img.filter(ImageFilter.FIND_EDGES)
    pixelsf = imgf.load()
    pixels = img.load()
    for i in range(imgf.size[0]):
        for j in range(imgf.size[1]):
            edge_alph = pixelsf[i,j][3]
            tup = (MAGIC - edge_alph,
                MAGIC - edge_alph,
                MAGIC - edge_alph,
                MAGIC*pixels[i,j][3])
            pixelsf[i,j] = tup
    return imgf


def get_derivatives(img: Image, ksize=9) -> Tuple[Image, Image]:
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    cvimg = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    sobelx = cv.Sobel(cvimg,cv.CV_64F,1,0,ksize=ksize)
    sobely = cv.Sobel(cvimg,cv.CV_64F,0,1,ksize=ksize)

    sobelx = (sobelx/np.max(sobelx)).mean(axis=-1)
    sobely = (sobely/np.max(sobely)).mean(axis=-1)
    outx = Image.new('RGBA', sobelx.shape)
    outy = Image.new('RGBA', sobely.shape)
    pixelsx = outx.load()
    pixelsy = outy.load()
    pixels = img.load()
    print(sobelx.shape, img.size, outx.size)
    for i in range(sobelx.shape[0]):
        for j in range(sobelx.shape[1]):
            dx = int(sobelx[i,j]*MAGIC - MAGIC/2)
            dy = int(sobely[i,j]*MAGIC - MAGIC/2)
            pixelsx[i,j] = (dx, dx, dx, pixels[j,i][3])
            pixelsy[i,j] = (dy, dy, dy, pixels[j,i][3])

    return outx, outy


def displacement_map(img: Image, map: Image, xdis: int = 10, ydis: int = 10):
    assert img.size == map.size
    res = Image.new('RGBA', img.size)
    res_pixels = res.load()
    pixels = img.load()
    map_pixels = map.load()
    for i in range(res.size[0]):
        for j in range(res.size[1]):
            mp = map_pixels[i,j]
            darkness = 1- (mp[0]+mp[1]+mp[2])*mp[3] / (3*MAGIC*MAGIC)
            x, y = (i - int(darkness*xdis)) % res.size[0], (j - int(darkness*ydis)) % res.size[1]
            res_pixels[i, j] = pixels[x, y]
    return res
