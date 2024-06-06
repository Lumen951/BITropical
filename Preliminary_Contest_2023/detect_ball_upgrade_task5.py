'''
日期:20240606
修改人员:流明Lumen
内容:标准赛踢小球
功能说明:去除图中直线和白色背景识别小球，返回小球圆心坐标
'''

import cv2
import numpy as np
import math

whiteLower = (0,0,130)
whiteUpper = (179,67,255)

# 输入值：图片，识别圆的最小半径范围和最大半径范围
def detect_Ball_upgrade(readimg, minRadius, maxRadius):
    
    # 最后要返回的坐标
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
    
    
    # 检测区            
    # cv2.imshow("removelineimg",removelineimg)
    # cv2.waitKey(0)     
    
    return ball
                

    


if __name__ == '__main__':
    
    # 这是用于储存坐标的txt,到时候要换成输入检测的代码
    file  = open("images/task5/detect_ball_upgrade.txt","w")
    
    for i in range(1,6):
        
        # 储存答案
        ans = []
        
        # 打开的图片
        filename = "images/task5/%d.jpg"%i
        
        # 读取图片
        readimg = cv2.imread(filename=filename)
        
        # 用于检测的图片，这些后面要删除
        testimg = readimg.copy()
        
        # 主检测程序
        ans = detect_Ball_upgrade(readimg, 0, 100)
        
        # print(ans)
        
        # 输入答案到文件中
        print(ans,file=file)
        
        # 检测程序
        for circle in ans:
            cv2.circle(testimg, (int(circle[0][0]), int(circle[0][1])), int(20), (0, 255, 0), thickness=2)
        
        cv2.imshow("circles",testimg)
        cv2.waitKey(2000)
        
        



