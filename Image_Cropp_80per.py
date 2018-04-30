from PIL import Image
#from multiprocessing import Pool
import os, os.path, sys
from os.path import isfile, join
from PIL import Image
from joblib import Parallel, delayed

path=os.getcwd()
dirs = os.listdir(path)
onlyfiles = [f for f in dirs if isfile(join(path, f))]
onlyimages = [f for f in onlyfiles if f.endswith('.jpeg')]

res = 2140

def crop(res):
    new_width=res
    new_height=res
    i = 0
    for item in onlyimages:

        fullpath = os.path.join(path,item)         #corrected
        if os.path.isfile(fullpath):
            img = Image.open(fullpath)
            width, height = img.size
            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2
            f, e = os.path.splitext(fullpath)
            imCrop = img.crop((left, top, right, bottom)) #corrected
            imCrop.save(f + '_resized.png', "png")
            #print('Image Cropped as '+ f + '_resized.jpeg')
            i+=1
            print("images left: "+str(len(onlyimages)-i))
    return result
crop(res)
result = Parallel(n_jobs=-1, verbose=1)(delayed(crop)(item) for item in onlyimages)

