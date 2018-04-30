import cv2
import numpy as np
import os, os.path, sys
from os.path import isfile, join
from PIL import Image

path=os.getcwd()
dirs = os.listdir(path)
onlyfiles = [f for f in dirs if isfile(join(path, f))]
onlyimages = [f for f in onlyfiles if f.endswith('.jpeg')]

gridsize=8
def histogram_normalization():
    i = 0
    for item in onlyimages:
        fullpath = os.path.join(path,item)         #corrected
        if os.path.isfile(fullpath):
            #img = cv2.imread(fullpath,1)
            #colorspace = cv2.cvtColor(img, cv2.COLOR_BGR2HLS) # For HLS
            #colorspace = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # For HSV#COLOR_BGR2HSV
            #colorspace = cv2.cvtColor(img, cv2.COLOR_BGR2LUV) # For #COLOR_BGR2LUV 
            bgr = cv2.imread(fullpath)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            lab_planes = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize)) # For CLAHE
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            # colorspace = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) # COLOR_BGR2YCrCb
            # colorspace[:,:,0] = cv2.equalizeHist(colorspace[:,:,0])
            # hst_out = cv2.cvtColor(colorspace, cv2.COLOR_YCrCb2BGR)
            f, e = os.path.splitext(fullpath)
            # cv2.imwrite(f+'_normd_luv.png',hst_out)
            cv2.imwrite(f+'_clahe.png',bgr)
            # f, e = os.path.splitext(fullpath)
            # imCrop = img.crop((left, top, right, bottom)) #corrected
            # imCrop.save(f + '_resized.png', "png")
            #print('Image Cropped as '+ f + '_resized.jpeg')
            i+=1
            print("images left: "+str(len(onlyimages)-i))
    return result
histogram_normalization()

# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# img = cv2.merge((b,g,r))
# res = np.hstack((img,equ)) #stackin images side-by-side
# equ = cv2.equalizeHist(hsv)
# print(type(equ))
# # print(type(res))
# cv2.imwrite('16_right_norm.jpeg',equ)

# from PIL import Image
# #from multiprocessing import Pool
# import os, os.path, sys
# from os.path import isfile, join
# from PIL import Image
# from joblib import Parallel, delayed

# path=os.getcwd()
# dirs = os.listdir(path)
# onlyfiles = [f for f in dirs if isfile(join(path, f))]
# onlyimages = [f for f in onlyfiles if f.endswith('.jpeg')]

# res = 2140

# def crop(res):
#     new_width=res
#     new_height=res
#     i = 0
#     for item in onlyimages:

#         fullpath = os.path.join(path,item)         #corrected
#         if os.path.isfile(fullpath):
#             img = Image.open(fullpath)
#             width, height = img.size
#             left = (width - new_width)/2
#             top = (height - new_height)/2
#             right = (width + new_width)/2
#             bottom = (height + new_height)/2
#             f, e = os.path.splitext(fullpath)
#             imCrop = img.crop((left, top, right, bottom)) #corrected
#             imCrop.save(f + '_resized.png', "png")
#             #print('Image Cropped as '+ f + '_resized.jpeg')
#             i+=1
#             print("images left: "+str(len(onlyimages)-i))
#     return result
# crop(res)
# result = Parallel(n_jobs=-1, verbose=1)(delayed(crop)(item) for item in onlyimages)

