#!/usr/bin/python2

import sys
import os
from tqdm import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

DATA_DIR = '/home/nico/Data/GalaxyZoo/'


def process_targets(redo=False):
    if not os.path.exists(os.path.join(DATA_DIR, 'targets.npy')) or redo is True:
        targets = np.genfromtxt(os.path.join(DATA_DIR, 'training_solutions_rev1.csv'),
                                delimiter=',', filling_values=0, skip_header=1)
        np.save(os.path.join(DATA_DIR, 'targets.npy'), targets[:, 1:])


def main():
    process_targets()

    finalSize = 128

    filenumbers = [[], []]
    for ii, dr in enumerate(['images_training_rev1', 'images_test_rev1']):
        for cnt, name in tqdm(enumerate(os.listdir(os.path.join(DATA_DIR, dr)))):
            number, ext = os.path.splitext(name)
            if os.path.isfile(os.path.join(DATA_DIR, dr, name)) and ext == '.jpg':
                filenumbers[ii].append(number)

                im = cv2.imread(os.path.join(DATA_DIR, dr, name))

                # IMAGE 1: the middle square of the original image
                # Note: I treat this as a noisy version of the image; presented only
                # during training to help generalization
                imraw = im[212 - finalSize / 2:212 + finalSize / 2, 212 - finalSize / 2:212 + finalSize/2]
                cv2.imwrite(os.path.join(DATA_DIR, dr[:-5] + '_proc', number + '_raw.png'), imraw)

                # IMAGE 2: a heavily preprocessed version of the original image
                # convert to grayscale
                imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                # threshold so that dark noise is removed
                _, imp = cv2.threshold(imgray, 31, 255, cv2.THRESH_TOZERO)
                # blur the image
                imp = cv2.medianBlur(imp, 5)
                imp = cv2.GaussianBlur(imp, (5, 5), 0)
                # adaptive histogram equalization
                clahe = cv2.createCLAHE(clipLimit=10., tileGridSize=(8, 8))
                imc = clahe.apply(imp)

                # Perform SURF, find keypoints and descriptors
                surf = cv2.SURF(hessianThreshold=2000, nOctaves=10)
                keyPoints, des = surf.detectAndCompute(imc, None)

                scores = []
                for kp in keyPoints:
                    dist = math.hypot(kp.pt[0] - 211, kp.pt[1] - 211)
                    if dist > 25:
                        scores.append(np.Inf)
                    else:
                        scores.append(dist - 0.1 * kp.size)

                if len(scores) == 0:
                    # no keypoint found in center, just cut out a centered box, unrotated, size 240
                    cntr = (211, 211)
                    rot = 0
                    diam = 256
                else:
                    cntr = keyPoints[np.argmin(scores)].pt
                    rot = keyPoints[np.argmin(scores)].angle
                    diam = max(64, keyPoints[np.argmin(scores)].size)

                # rotate, scale and crop the original (not grayscale) image
                rot_mat = cv2.getRotationMatrix2D(cntr, rot + 90, (finalSize * 3/4) / diam)
                imr = cv2.warpAffine(im, rot_mat, list(im.shape).append(1), flags=cv2.INTER_CUBIC)
                newimg = imr[212 - finalSize / 2:212 + finalSize / 2, 212 - finalSize / 2:212 + finalSize/2, :]

                # plot
                # im2 = cv2.drawKeypoints(im, keyPoints, None, (255, 0, 0), 4)
                # if cnt < 50:
                #     res = np.hstack((im, cv2.cvtColor(imc, cv2.COLOR_GRAY2BGR), im2))
                #     plt.imshow(res), plt.show()
                #     plt.imshow(newimg), plt.show()

                cv2.imwrite(os.path.join(DATA_DIR, dr[:-5] + '_proc', number + '_proc.png'), newimg)

    # sort, so that targets/outputs and filenumbers are aligned
    [x.sort() for x in filenumbers]
    np.save(os.path.join(DATA_DIR, 'filenumbers.npy'), filenumbers)


if __name__ == '__main__':
    sys.exit(main())
