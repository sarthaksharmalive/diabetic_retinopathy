from PIL import Image
#from multiprocessing import Pool
import os.path, sys
    
path = "/run/media/sarthak/CAF44438F44428D1/dataset/train/train"
dirs = os.listdir(path)
source_file = "/run/media/sarthak/CAF44438F44428D1/dataset/filesToBeResized"
image_list_file = open(source_file, "r")
each_file_list = image_list_file.read().split('\n')


new_width=1700
new_height=1700
def crop():
    i = 0
    for item in each_file_list:

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
            print("images left: "+str(len(each_file_list)-i))

crop()