
import cv2 
import numpy as np
import math

greenLower = (35,0,109)
greenUpper = (150,71,255)

def detect_pit(readimg):
    
    copyimg = readimg.copy()
    
    # 利用高斯滤波的线性，这几个部分前后顺序可以调换
    blurImg = cv2.GaussianBlur(readimg, (11,11), 0)
    
    # 转换为hsv色彩空间
    hsvImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)
    
    # 获取二值图
    mask1 = cv2.inRange(hsvImg, greenLower, greenUpper)

    # 去除一些杂乱区域（地板的花边）
    kernel = np.ones((5, 5), np.uint8)
    mask2 = cv2.erode(mask1, kernel)
    mask3 = cv2.dilate(mask2, kernel, iterations=1)

    # 获得图像边缘坐标
    edge = cv2.Canny(mask3, 40, 80)
    # cv2.imshow("orgedge",edge)
    
    # 这里contours是一个二维数组，需要进行转化输入boundingRect
    contours , hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 通过判断矩形的大小，去除雪花噪点
    x,y,w,h = 0,0,0,0
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        if w*h < 10000 or w*h > 100000:
            cv2.drawContours(edge, [contours[i]], -1, (0,0,0), thickness=1)

    # 取新的边缘进行计算，将过去的边缘去除
    contours , hierarchy = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
    
    '''
    # 用于看清的干不干净的代码
    x,y,w,h = 0,0,0,0
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        cv2.rectangle(copyimg, (x,y), (x+w,y+h), (0,0,255), thickness=2)
    '''
    
    # 这里没取反，首先反着取，之后调整
    # 命名的逻辑是 max_x max_y代表最右侧的点(因为手头图片都是最大值在右侧) min_x min_y对应左边的点，因为x最小值对应桥头最左边的点
    max_x = x
    max_y = y
    min_x = x + w
    min_y = y + h
    
    # contours存储方式：四维list 第一个值是轮廓的数，也就是第几个轮廓 第二个值是第几个点 第三个和第四个值分别是x和y
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            # max对应最右边的点 min对应最左边的点
            max = np.max(contours[i][j], axis=0)
            min = np.min(contours[i][j], axis=0)
            if max[1] + max[0] >= max_y + max_x:
                max_x = max[0]
                max_y = max[1]
                
            if min[1] - min[0] >= min_y - min_x:
                min_x = min[0]
                min_y = min[1]  # 这一步没错！这是要取那一个contour组里面y的对应的值

    # 画条线检查一下
    cv2.line(copyimg, (min_x,min_y) , (max_x,max_y), (255,0,0), 3)
    
    cv2.imshow("edge",edge)
    cv2.imshow("readimg",copyimg)
    cv2.waitKey(0)
    
    
    
if __name__ == "__main__":
    
    
    for i in range(1,6):
        
        readimg = cv2.imread("images2/task3/%d.jpg"%i)
        
        detect_pit(readimg)
    





