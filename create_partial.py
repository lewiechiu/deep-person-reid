import cv2
import numpy as np
from PIL import Image
import os
from IPython.display import display

cnt = 0
for root, d, files in os.walk("../Market-1501-v15.09.15/market1501/Market-1501-v15.09.15/query/"):
    for file in files:
        if ".jpg" not in file:
            continue
        location = os.path.join(root, file)
        save_loc = os.path.join("/".join([*root.split("/")[:-2], "query_occlusion"]) , file)
#         print(save_loc)
#         print(location)
        im = Image.open(location)
        a= np.random.randint(0, 128)
        tl = np.random.randint(0,1 )
        occlusion = Image.new('RGB', (64, 32))
        im.paste(occlusion, (tl, a))
        im.save(save_loc)
#         display(im)
        cnt += 1
#         if cnt == 10:
#             break