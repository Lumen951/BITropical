
# -*- coding: utf-8 -*-

'''
日期:20240606
修改人员:流明Lumen
内容:初赛所有题目合并版-初版
待升级点:程序鲁棒性,尤其是HSV色彩范围的鲁棒性。
       其他鲁棒性已经在函数上方标出。
待优化点:添加关卡筛选器,即若将图片全存储在同一个文件夹里,如何通过筛选使用函数识别并返回值
'''

import cv2 
import numpy as np
import os
import math

class Detect:
    
    def __init__(self, readimg):
        self.readimg = readimg

    # 用于解决task1的球检测(有待升级,也许可以同最后一问的程序进行合并,或者更换检测方法)
    def ball_task1(self, readimg):
        # 进行高斯模糊处理
        gauss_blur = cv2.GaussianBlur(readimg, (5,5), 0)
        
        # 进行膨胀运算
        dilate = cv2.dilate(gauss_blur, (3,3), 1)
        
        # Canny边缘检测
        canny = cv2.Canny(dilate, 40, 80)
        
        # 霍夫圆检测
        circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 30, param1=80, param2=30, minRadius=50, maxRadius=200)
        
        # 返回值
        if circles is None:
            return 0, 0, 0
        else:
            return circles[0, 0, 2], circles[0, 0, 0], circles[0, 0, 1]

    # 用于解决task2的桥检测(升级点在于检测的速度和鲁棒性，精确度没有问题)
    def bridge(self, readimg):
        
        bridge = list()
        
        # 设置HSV阈值
        # 注意:这里的HSV阈值应设定较大范围，以提高鲁棒性
        greenLower = (65, 19, 74)
        greenUpper = (117, 255, 135)
        
        # 利用高斯滤波的线性，这几个部分前后顺序可以调换
        blurImg = cv2.GaussianBlur(readimg, (11,11), 0)
        
        # 转换为hsv色彩空间
        hsvImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)
        
        # 获取二值图
        binaryimg = cv2.inRange(hsvImg, greenLower, greenUpper)

        # 去除一些杂乱区域（地板的花边）
        kernel = np.ones((5, 5), np.uint8)
        binaryimg = cv2.erode(binaryimg, kernel)
        binaryimg = cv2.dilate(binaryimg, kernel, iterations=1)

        # 获得图像边缘坐标
        edge = cv2.Canny(binaryimg, 40, 80)
        
        # 这里contours是一个二维数组，需要进行转化输入boundingRect
        contours , hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 通过判断矩形的大小，去除雪花噪点
        x,y,w,h = 0,0,0,0
        for i in range(len(contours)):
            x,y,w,h = cv2.boundingRect(contours[i])
            if w*h < 1000:
                cv2.drawContours(edge, [contours[i]], -1, (0,0,0), thickness=1)

        # 取新的边缘进行计算，将过去的边缘去除
        contours , hierarchy = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
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

        # 计算两点距离
        distance = math.sqrt(pow((max_x-min_x), 2) 
                             + pow((max_y-min_y), 2))
        
        bridge = [min_x , min_y, max_x, max_y, distance]
        
        return bridge

    # 用于解决task3的坑检测(升级点在于检测的速度和鲁棒性，尤其是坑的大小和角度发生变化后的鲁棒性)
    def pit(self, readimg):
        
        pit = list()
        
        greenLower = (35,0,109)
        greenUpper = (150,71,255)
        
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

        distance = math.sqrt((max_x - min_x) ** 2 
                             + (max_y - min_y) ** 2)
        
        pit = [min_x,min_y,max_x,max_y,distance]
        
        return pit

    # 用于解决task4的门检测(升级点在于去除背后块以及精度,提取contours的位置不准)
    def door(self, readimg):
        
        purple_Lower = (87,25,30)
        purple_Upper = (147,255,255)
        
        door = list()
        
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

        door = (x_l, y_l, x_r, y_r)
        
        return door

    # 用于解决task5的球检测(基本没有升级点,鲁棒性和检测精度都很好)
    def ball_task5(self, readimg, minRadius, maxRadius):
        
        whiteLower = (0,0,130)
        whiteUpper = (179,67,255)
        
        # 最终答案
        ball = list()
    
        # 常规操作，变灰度图，然后再模糊
        hsvimg = cv2.cvtColor(readimg, cv2.COLOR_BGR2HSV)
        blurimg = cv2.GaussianBlur(hsvimg, (5,5), 0)
        
        # 二值化操作
        binaryimg = cv2.inRange(blurimg, whiteLower, whiteUpper)
        
        # 这一个kernel用于去除地上的白线
        # 白线有横有纵，所以用11*11的kernel进行去除
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))
        
        # 去除地上的白线
        removelineimg = cv2.morphologyEx(binaryimg, cv2.MORPH_OPEN, kernel1)
        
        # 这一个kernel用于去除各种杂点
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3), (-1,-1))
        
        # 这两步用于去除杂点，让图像更干净
        removelineimg = cv2.morphologyEx(removelineimg, cv2.MORPH_CLOSE, kernel2, (-1,-1))
        removelineimg = cv2.morphologyEx(removelineimg, cv2.MORPH_OPEN, kernel2, (-1,-1))
        
        # 找边缘
        contours, hierarchy = cv2.findContours(removelineimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 用contours进行筛选
        for cnt in contours:
            
            # 接下来的每一步都是筛选
            
            # len(cnt)代表cnt中点的个数，这一步是为了筛选过少的点连起来的杂区域
            # 本质也利用了contours的存储结构，也就是存储轮廓顺序，之后存储点的序号，然后是xy值
            # 如果点的序号比5个还少（5个看似很少，其实是因为SIMPLE模式下会压缩一些点的存储，比如一条直线只会保留首尾两个点）
            # 那就说明是个杂区域
            if len(cnt) < 5:
                continue
            
            # 筛选适合的区域，也就是只要某个半径面积范围内的区域
            area = cv2.contourArea(cnt)
            if area < (minRadius**2) * math.pi or area > (maxRadius**2) * math.pi:
                continue
            
            # 计算轮廓cnt的周长
            arc_length = cv2.arcLength(cnt, True)
            
            # 算出区域的周长
            radius = arc_length / (2 * math.pi)
            
            # 通过周长反算出半径后，卡掉多余的圆
            if not (minRadius < radius and radius < maxRadius):
                continue
            
            # 用于拟合轮廓为一个椭圆，为后续检测长宽比进行参考
            ellipse = cv2.fitEllipse(cnt)
            
            # 计算长宽比，也就是a/b的大小
            ratio = float(ellipse[1][0]) / float(ellipse[1][1])
            
            # 如果这个长宽比在0.9-1.1之间的话，就说明这个区域十分接近圆形，可以进行角点的计算
            # 这里可能有一点bug也就是偏相机边缘的部分会因为镜头的变形而导致识别不出来
            # 所以将比值调试为0.7-1.1
            if ratio > 0.7 and ratio < 1.1:
                
                # 为什么要找角点？因为现在还有可能识别出来是正方形
                corner = cv2.approxPolyDP(cnt, 0.02 * arc_length, True)
                cornerNum = len(corner)
                
                # 如果顶点数大于4个就被认为是圆形
                if cornerNum > 4:
                    ball.append(ellipse)
        
        return ball
        
        
if __name__ == '__main__':
    
    path = "images/"
    
    dirs = os.listdir(path)
    
    for file in dirs:
        
        ans = []
        
        # 对于task1进行检测并输出答案
        if file == "task1":
            
            images = os.listdir(path+file+'/')
            
            for img in images:
                
                if img.endswith(".jpg"):
                    readimg = cv2.imread(path+file+'/'+img)
                    
                    detect = Detect(readimg)
                    
                    ans = detect.ball_task1(readimg)
                    
                    print(ans)
        
        if file == "task2":
            
            images = os.listdir(path+file+'/')
            
            for img in images:
                
                if img.endswith(".jpg"):
                    readimg = cv2.imread(path+file+'/'+img)
                    
                    detect = Detect(readimg)
                    
                    ans = detect.bridge(readimg)
                    
                    print(ans)

        if file == "task3":
            
            images = os.listdir(path+file+'/')
            
            for img in images:
                
                if img.endswith(".jpg"):
                    readimg = cv2.imread(path+file+'/'+img)
                    
                    detect = Detect(readimg)
                    
                    ans = detect.pit(readimg)
                    
                    print(ans)
                    
        if file == "task4":
            
            images = os.listdir(path+file+'/')
            
            for img in images:
                
                if img.endswith(".jpg"):
                    readimg = cv2.imread(path+file+'/'+img)
                    
                    detect = Detect(readimg)
                    
                    ans = detect.door(readimg)
                    
                    print(ans)
                    
        if file == "task5":
            
            images = os.listdir(path+file+'/')
            
            for img in images:
                
                if img.endswith(".jpg"):
                    readimg = cv2.imread(path+file+'/'+img)
                    
                    detect = Detect(readimg)
                    
                    ans = detect.ball_task5(readimg, 0, 100)
                    
                    print(ans)
                
                
            
            


