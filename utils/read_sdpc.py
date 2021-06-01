import io
import os
import sys
from io import BytesIO, StringIO

import cv2
import matplotlib.pyplot as plt
from PIL import Image

path_sdpc = 'samples/20210215_145052.sdpc'

# type(f)<class '_io.BufferedReader'>
f = open(path_sdpc, 'rb')

# type(img)<class 'bytes'>
f_bytes = f.read()

img = Image.frombytes("RGB", (1000, 1000), bytes(f_bytes))

img.save('output/test.png')
print('demo')
