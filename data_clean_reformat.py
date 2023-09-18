#Photo format and name normalization (e.g. all to png/jpg)
import os
import sys
import cv2
import numpy as np

def listfiles(rootDir,rename=False):
    list_dirs = os.walk(rootDir)
    num = 0
    for root, dirs, files in list_dirs:
        for d in dirs:
            print(os.path.join(root,d))
        for f in files:
            fileid = f.split('.')[0] 
            filepath = os.path.join(root,f)
            try:
                src = cv2.imread(filepath,1)
                print("src=",filepath,src.shape)
                os.remove(filepath) #delete original photo
                if rename:
                    cv2.imwrite(os.path.join(root,str(num)+".jpg"),src) #write and rename new image
                    num = num + 1
                else:
                    cv2.imwrite(os.path.join(root,fileid+".jpg"),src) #write new image
            except:
                os.remove(filepath) #remove broken images
                continue

listfiles(sys.argv[1],rename=True)