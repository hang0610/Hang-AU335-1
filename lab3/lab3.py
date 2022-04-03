import cv2
import numpy as np

src = cv2.imread("./img.png", 0)
origin = cv2.imread("./img.png")
src = cv2.resize(src,(1080, 720))
origin = cv2.resize(origin,(1080, 720))
# 高斯滤波
img_blur=cv2.GaussianBlur(src,(5,5),5)

# 自适应阈值分割
# img_thresh=cv2.adaptiveThreshold(src,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,3)
img_thresh=cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,3)
cv2.imshow('img_thresh_blur', img_thresh)

# 初步膨胀运算
kernel1 = np.ones((3,3),np.uint8)
dilation1 = cv2.dilate(img_thresh,kernel1,iterations = 2)
cv2.imshow('dilation1', dilation1)

# 将单词那一行全部填充
col = dilation1.shape[1]
horizontalKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (col, 1))
dilation2 = cv2.dilate(dilation1,horizontalKernel,iterations = 1)
cv2.imshow('dilation2', dilation2)

# 腐蚀运算
kernel2 = np.ones((4,4),np.uint8)
erode = cv2.erode(dilation2, kernel2, iterations = 8)
cv2.imshow('erode', erode)

# 霍夫直线检测
# 这里取变化弧度为pi/2，则经过图片预处理检测出的直线一定为水平直线，即极角为pi/2
lines = cv2.HoughLines(erode, 0.9, np.pi/2, 300)
# print(lines.ndim)
# print(lines.shape)

# 之后需要将检测到的直线通过极坐标的方式画出来
# 此时需要考虑所画删除线的线段起终点
black_img = np.zeros([720, 1080, 3], dtype=np.uint8)
for line in lines: # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的，theta是弧度
    rho, theta = line[0] # 下述代码为获取 (x0,y0) 具体值
    y0 = int(rho)
    img_line = img_thresh[y0,:].tolist()
    x1 = 0
    x2 = 1080
    y1 = y2 = y0
    for i in range(1080):
        if img_line[i] == 0:
            continue
        else:
            x1 = i
            break
    for j in range(1080):
        if img_line[1079-j] == 0:
            continue
        else:
            x2 = 1080-j
            break
    cv2.line(black_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

# 将同一行内容的多条稀疏删除线在竖直方向上合并
verticalKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
black_mask = cv2.dilate(black_img,verticalKernel,iterations = 3)
cv2.imshow('black_mask', black_mask)

# 竖直方向上腐蚀删除线，细化删除线优化可视化效果
kernel3 = np.ones((2,2),np.uint8)
black_mask = cv2.erode(black_mask, kernel3, iterations=5)

# 将black_mask黑白值颠倒，将删除线置为黑色
white_mask = ~black_mask
cv2.imshow("white_img", white_mask)

# 将white_mask与原图进行与运算，实现在原图上画出黑色的删除线
final = cv2.bitwise_and(origin, white_mask)
cv2.imshow("final", final)
cv2.imwrite('result.png', final)
while True:
    q = cv2.waitKey(1)
    if q == ord('q'):
        cv2.destroyAllWindows()
        break