
import cv2
import numpy as np
import math

purple_Lower = (87,25,30)
purple_Upper = (147,255,255)

# 问题依然在如何提取坐标，有了轮廓，不知道怎么找到坐标
def detect_door(readimg):
    
    copy_img = readimg.copy()
    
    hsvimg = cv2.cvtColor(readimg, cv2.COLOR_BGR2HSV)
    
    blurimg = cv2.GaussianBlur(hsvimg, (3,3), 0)
    
    binaryimg = cv2.inRange(blurimg, purple_Lower, purple_Upper)
    
    # 两次去除，一次去除背景的横线，一次去除底边的地板花纹
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(1,11))
    
    removelineimg = cv2.morphologyEx(binaryimg, cv2.MORPH_OPEN, kernel1)
    
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3), (-1,-1))
    
    removelineimg = cv2.morphologyEx(removelineimg, cv2.MORPH_CLOSE, kernel2, (-1,-1))
    removelineimg = cv2.morphologyEx(removelineimg, cv2.MORPH_OPEN, kernel2, (-1,-1))
    
    edge = cv2.Canny(removelineimg, 40, 80)
    
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    x,y,w,h = 0,0,0,0
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        if w*h < 5000:
            removelineimg[y: y+h, x: x+w] = 0
            cv2.drawContours(edge, [contours[i]], -1, (0,0,0), thickness=1)
            
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    
    # 分别代表左边的点的坐标和右边的点的坐标
    x_l, y_l, x_r, y_r = 0, 0, 0, 0
    
    # flag用于标记是第一个轮廓还是第二个轮廓
    flag = 1
    
    for i in range(len(contours)):
        if len(contours[i]) > 5:
            for j in range(len(contours[i])):
                max = np.max(contours[i][j], axis=0)
                if flag == 1:
                    if max[1] >= y_r:
                        y_r = max[1] 
                        x_r = max[0]
                if flag == 2:
                    if max[1] >= y_l:
                        y_l = max[1] 
                        x_l = max[0]
            flag += 1
            print(max)

    ans = (x_l, y_l, x_r, y_r)
    
    print(ans)
    
    cv2.line(copy_img, (x_l, y_l), (x_r, y_r), (0,0,255), thickness=2)
    
    '''
    # 已废弃方法
    # 使用LSD直线检测,效果比霍夫直线检测要好
    LSD = cv2.createLineSegmentDetector(0)
    
    lines = LSD.detect(removelineimg)
    
    for line in lines[0]:
        x0 = int(round(line[0][0]))
        y0 = int(round(line[0][1]))
        x1 = int(round(line[0][2]))
        y1 = int(round(line[0][3]))
        cv2.line(copy_img, (x0, y0), (x1, y1), (0,255,0), thickness=2)
    '''
    # 测试区
    cv2.imshow("resourceimg",copy_img)
    
    # cv2.imshow("removelineimg", removelineimg)
    
    # cv2.imshow("edge",edge)
    
    cv2.waitKey(0)
    
    return ans
    


if __name__ == '__main__':
    
    file = open("images2/task4/detect_door.txt",'w')
    
    print("x_l y_l x_r y_r", file=file)
    
    for i in range(1,6):
    
        ans = []

        readimg = cv2.imread("images2/task4/%d.jpg"%i)
    
        ans = detect_door(readimg)
        
        print(ans, file=file)
