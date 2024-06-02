import cv2
import numpy as np

def detect_ball(readimg):
    # 灰度图转换
    # gray = cv2.cvtColor(readimg, cv2.COLOR_BGR2GRAY)
    
    # 进行高斯模糊处理
    gauss_blur = cv2.GaussianBlur(readimg, (5,5), 0)
    
    # 进行膨胀运算
    dilate = cv2.dilate(gauss_blur, (3,3), 1)
    
    # Canny边缘检测
    canny = cv2.Canny(dilate, 40, 80)
    
    # 霍夫圆检测
    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 30, param1=80, param2=30, minRadius=50, maxRadius=200)

    # 观察区
    # cv2.imshow("canny",canny)
    # cv2.imshow("dilate",dilate)
    
    # 看一下怎么存储的
    # print(circles)
    
    cv2.waitKey(0)
    
    # 返回值
    if circles is None:
        return 0, 0, 0
    else:
        return circles[0, 0, 2], circles[0, 0, 0], circles[0, 0, 1]

if __name__ == "__main__":
    
    # 打开已创建文件
    file = open("images2/task1/detect_ball.txt",'w')
    
    # 在第一行先打个r,x,y
    print("r x y",file=file)
    for i in range(1,8):
        
        ans = []
        
        # 遍历读取文件
        filename = "images2/task1/%d.jpg"%i
        
        # 读取图片
        readimg = cv2.imread(filename=filename)
        
        # 使用函数
        ans = detect_ball(readimg)
        
        #画圆看一眼
        r,x,y = int(ans[0]), int(ans[1]), int(ans[2])
        cv2.circle(readimg, (x, y), r, (255, 0, 0), 1)
        
        cv2.imshow("circle", readimg)
        cv2.waitKey(0)
        
        # 写入文件
        print(ans,file=file)
    
    # 关闭文件
    file.close()
        
        
        
    







