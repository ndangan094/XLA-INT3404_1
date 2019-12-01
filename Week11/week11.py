import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math


row = 6
col = 6
block = row * col

def suffle(img, x, y):
    img2 = np.array(img)
    img2 = img2[0:row*y, 0:col*x, 0:3]

    block_img = np.zeros((y, x, 3, block))
    suffle_img = np.zeros((row*y, col*x, 3), dtype='uint8')
    idx = np.random.permutation(block)

    for i in range(0,row):
        for j in range(0,col):
            minY = i * y
            minX = j * x
            tempidx = idx[i * row + j]
            block_img[:, :, :, tempidx] = img2[minY:minY + y, minX:minX + x, :]

            yy = math.floor(tempidx/row)
            xx = tempidx - yy * row
            suffle_img[yy * y: (yy + 1) * y, xx*x:(xx+1)*x, ::-1] = block_img[:, :, :, tempidx]
    return [suffle_img, block_img]



img = Image.open('./input.jpg')
[width,height] = img.size
[x,y] = [math.floor(width/col), math.floor(height/row)]
blocks = np.zeros((y, x, 3, col * row))
copyImg = np.array(img)

for i in range(row):
    for j in range(col):
        xx = j * x
        yy = i * y
        blocks[:, :, :, row * i + j] = copyImg[yy:yy +y, xx:xx+x, :]

[suffle_img, block_img] = suffle(img, x, y)

img = np.array(img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

threshold = 0.85
swap_list = []
swapped = [0] * row * col

for i in range(block):
    temp = block_img[:, :, :, i].astype('uint8')
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    w, h = temp.shape[::-1]

    thresh_list = []
    for j in range(block):
        if swapped[j] == 1:
            continue
        sourceImg = blocks[:, :, :, j].astype('uint8')
        sourceImg = cv2.cvtColor(sourceImg, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(sourceImg, temp, cv2.TM_CCOEFF_NORMED)
        if res >= threshold:
            thresh_list.append([j, res[0][0]])
    thresh_list = np.array(thresh_list)
    max_index = int(thresh_list[np.argmax(thresh_list[:, 1])][0])
    swap_list.append([i, max_index])
    swapped[max_index] = 1

block_images_solved = np.zeros((y, x, 3, block))
for swap in swap_list:
    block_images_solved[:, :, :, swap[1]] = block_img[:, :, :, swap[0]]
solved_image = np.zeros((height, width, 3), dtype='uint8')
for i in range(row):
    for j in range(col):
        xx = j * x
        yy = i * y
        block_index = row * i + j
        solved_image[yy:yy+y, xx:xx+x,::-1] = block_images_solved[:, :, :, block_index]

cv2.imwrite('suffle_img.jpg', suffle_img)
cv2.imwrite('solved_img.jpg', solved_image)