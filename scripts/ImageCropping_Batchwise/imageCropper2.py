from PIL import Image
#from multiprocessing import Pool
import os.path, sys
    
path = "/run/media/sarthak/CAF44438F44428D1/dataset/train/train"
dirs = os.listdir(path)
b1 = dirs[0:len(dirs)//10]
b2 = dirs[len(dirs)//10:2*len(dirs)//10]
b3 = dirs[2*len(dirs)//10:3*len(dirs)//10]
b4 = dirs[3*len(dirs)//10:4*len(dirs)//10]
b5 = dirs[4*len(dirs)//10:5*len(dirs)//10]
b6 = dirs[5*len(dirs)//10:6*len(dirs)//10]
b7 = dirs[6*len(dirs)//10:7*len(dirs)//10]
b8 = dirs[7*len(dirs)//10:8*len(dirs)//10]
b9 = dirs[8*len(dirs)//10:9*len(dirs)//10]
b10 = dirs[9*len(dirs)//10:10*len(dirs)//10]



new_width=1700
new_height=1700
def crop():
    i = 0
    for item in b2:

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
            imCrop.save(f + '_resized.jpeg', "jpeg", quality=100)
            #print('Image Cropped as '+ f + '_resized.jpeg')
            i+=1
            print("images left: "+str(len(b2)-i))

crop()