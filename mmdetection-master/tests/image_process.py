import os
from tqdm import tqdm
import cv2
from skimage import io
# libpng warning: iCCP: known incorrect sRGB profile 警告，问题解决
path = r"/liu/icme/dataset/train/images/"
fileList = os.listdir(path)
for i in tqdm(fileList):
    image = io.imread(path+i)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imencode('.jpg',image)[1].tofile(path+i)

