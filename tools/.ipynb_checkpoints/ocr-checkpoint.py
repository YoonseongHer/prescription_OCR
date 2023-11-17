import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFont,Image,ImageDraw
from tools.preprocessing import crop_box

reader = easyocr.Reader(['ko','en'])

def OCR_result(img):
    img = cv2.imread(img).copy() if type(img)==str else img.copy()

    detect_img = np.ones(img.shape)*255
    fontpath = "/usr/share/fonts/truetype/nanum/NanumSquareR.ttf"
    font = ImageFont.truetype(fontpath, 40)
    img_pil = Image.fromarray(detect_img.astype('uint8'))
    draw = ImageDraw.Draw(img_pil)
    b,g,r,a = 0,0,0,0
    
    result = reader.readtext(img)

    for box,text,confidence in result:
        if confidence >= 0.85:
            color = [0,255,0]
        elif confidence >= 0.1:
            color = [0,0,255]
        else:
            continue
        x,y = list(map(int,box[0])),list(map(int,box[2]))
        img = cv2.rectangle(img,x,y,color,2)
        font_color = color
        font_color[1]=0
        draw.text(x,text, font=font, fill=tuple(font_color))
    detect_img = np.array(img_pil)
    return img, detect_img, result