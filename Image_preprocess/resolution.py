import os
from os.path import isfile, join
from PIL import Image

resolution=[]
resolution_u=[]
path='/run/media/sarthak/CAF44438F44428D1/dataset/train_original'
dirs = os.listdir(path)
onlyfiles = [f for f in dirs if isfile(join(path, f))]
onlyimages = [f for f in onlyfiles if f.endswith('.jpeg')]
for images in onlyimages:
    fullpath = os.path.join(path,images)
    if os.path.isfile(fullpath):
        img = Image.open(fullpath)
        width, height = img.size
        wxh=(width,height)
        resolution.append(wxh)

print("Resolution "+str(len(resolution)))
resolution_u=set(resolution)
print("Unique Resolution "+str(len(resolution_u)))