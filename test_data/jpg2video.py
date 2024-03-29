import cv2
import glob

img_array = []
size = (0, 0)
for filename in glob.glob("pltoutput/*"):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter('pltoutput.avi',cv2.VideoWriter_fourcc(*'DIVX'), 8, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
