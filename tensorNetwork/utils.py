import numpy as np
import cv2
from functools import reduce
import math

def image_process(path='img1.png', resize=(256, 256), shape=(16, 16, 16, 16, 3)):
    img = cv2.imread(path)
    #print(img.shape)
    #img = cv2.resize(img, resize)
    #cv2.imwrite(path, img)
    #img = img / 255 if img.max()>1 else img
    ten = img.reshape(shape)
    return ten

def calPara(shape):
    return reduce(np.multiply, shape, 1)

def calParaGraph(graph):
    para = 0
    for i in range(len(graph)):
        core = np.concatenate([graph[0:i, i], graph[i, i:]], axis=0)
        para += calPara([i for i in core if i != 0])
    return para
