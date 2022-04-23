import numpy as np
from PIL import Image

src = np.random.rand(20)
# src = np.pad(src, (2, 2), 'constant', constant_values=(0, 0))
print(src)
print(np.size(src))

src_img = Image.fromarray(src)
print(src_img.size)
src_img_padding = src_img.resize((1, 10), Image.ANTIALIAS)

dst = np.zeros(10)
for i in range(src_img_padding.size[1]):
    dst[i] = src_img.getpixel((0, i))
print(dst)