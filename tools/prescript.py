import cv2
import numpy as np

def cell_detect(img_url):
    src = cv2.imread(img_url)
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10, blockSize=3, useHarrisDetector=True, k=0.03)
    corners = corners.astype(int)
    canny = cv2.Canny(src,100,300)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(canny, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(imgThreshold,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    t_list = []
    
    for i in range(len(contours)):
        contour = contours[i]
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        srcc = box
        srcc = np.where(srcc<0,0,srcc)

        con_size = [cv2.contourArea(contour) for contour in contours]
        max_index = con_size.index(max(con_size))
        if (hierarchy[0, i, 2] == -1 and hierarchy[0, i, 3] == max_index):
            t_list.append(box)
    t_list = [t.tolist() for t in t_list]
    return t_list

def cell_diagnal(cell):
    x_list,y_list = [],[]
    for point in cell:
        x_list.append(point[0])
        y_list.append(point[1])
    cell_d = [[min(x_list),min(y_list)],
              [max(x_list),max(y_list)]]
    return cell_d

def box_in(standard, target):
    '''
    staticmethod and target : [[x0,y0],[x1,y1]]
        0 : left top
        1 : right bottom
    '''
    (st_x0, st_y0), (st_x1, st_y1) = standard
    (tr_x0, tr_y0), (tr_x1, tr_y1) = target
    margin = 20
    
    if (st_x0-margin <= tr_x0) and (st_y0-margin <= tr_y0) and (st_x1+margin >= tr_x1) and (st_y1+margin >= tr_y1):
        return True
    else:
        return False
    
def cell_text(cell_list, ocr_result):
    result_boxs, result_texts = [], []
    cell_list = [cell_diagnal(cell) for cell in cell_list]
    for result in ocr_result:
        result_boxs.append(cell_diagnal(result[0]))
        result_texts.append(result[1])
        
    box_texts = []
    for box in cell_list:
        text = ''
        for t_box, t_text in zip(result_boxs, result_texts):
            if box_in(box, t_box):
                text += t_text+' '
        box_texts.append(text)
    return box_texts