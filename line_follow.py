import cv2
import numpy as np
from ShowProcess import ShowProcess
import glob
import time
import matplotlib.pyplot as plt


plt.figure()


def func_time(func):
    def ft():
        s = time.time()
        func()
        e = time.time()
        print('use time :', e - s)
    return ft()


def get_contour_center(contour):
    m = cv2.moments(contour)

    if m["m00"] == 0:
        return [0, 0]

    x = int(m["m10"] / m["m00"])
    y = int(m["m01"] / m["m00"])

    return [x, y]


def process(image):
    _, thresh = cv2.threshold(
        image, 125, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(image, [main_contour], -1, (150, 150, 150), 2)
        contour_center = get_contour_center(main_contour)
        cv2.circle(image, tuple(contour_center), 2, (150, 150, 150), 2)
        return image, contour_center
    else:
        return image, (0, 0)


def slice_out(im, num):
    cont_cent = list()

    height, width = im.shape[:2]
    sl = int(height / num)
    sliced_imgs = list()
    for i in range(num):
        part = sl * i
        crop_img = im[part:part + sl, 0:width]
        processed = process(crop_img)

        plt.subplot2grid((4, 4), (i, 2), colspan=2, rowspan=1)
        plt.axis('off')
        plt.imshow(processed[0], cmap="gray")
        plt.title(str(i) + ", " + str(processed[1]))

        sliced_imgs.append(processed[0])
        cont_cent.append(processed[1])
    return sliced_imgs, cont_cent


def repack(images):
    im = images[0]
    for i in range(len(images)):
        if i == 0:
            im = np.concatenate((im, images[1]), axis=0)
        if i > 1:
            im = np.concatenate((im, images[i]), axis=0)
    return im


@func_time
def main():
    img_in_root = 'test_data/image/'
    img_out_root = 'test_data/pltoutput/'
    img_count = len(glob.glob(img_in_root + '*jpg'))
    process_bar = ShowProcess(img_count, 'OK!')
    no_slice = 4

    for x in range(img_count):
        img = cv2.imread(img_in_root + str(x).zfill(3) + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
        plt.axis('off')
        plt.imshow(img, cmap="gray")
        plt.title("Gray")

        slices, cont_cent = slice_out(img, no_slice)
        img = repack(slices)

        plt.subplot2grid((4, 4), (2, 0), colspan=2, rowspan=2)
        plt.axis('off')
        plt.imshow(img, cmap='gray')
        plt.title("Finish")

        plt.savefig(img_out_root + str(x).zfill(3) + '.jpg')
        # cv2.imwrite(img_out_root + str(x).zfill(3) + '.jpg', img)
        process_bar.show_process()


if __name__ == '__main__':
    main
