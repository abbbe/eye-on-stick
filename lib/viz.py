from PIL import Image, ImageDraw
from IPython import display
from io import BytesIO

import numpy as np

def showarray(img_array):
    buf = BytesIO()
    Image.fromarray(np.uint8(img_array)).save(buf, 'png')
    display.display(display.Image(data=buf.getvalue()))
