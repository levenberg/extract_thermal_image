#!/usr/bin/env python3
import rospy
import cv2
import os
import time
import random
import numpy as np


SHOW_RESULT_IMAGE = False
SHOW_DEBUG_IMAGE = False

# source images path and output path. change it
# IMAGE_PATH = "/home/huai/Documents/data/cloud_models/japan_bridge/thermal/image/"
# OUTPUT_PATH = "/home/huai/Documents/data/cloud_models/japan_bridge/thermal/corrected/"

# the pitch size of average around matched points
KERNEL_SIZE = 15
HALF_KERNEL_SIZE = int(KERNEL_SIZE/2)



# image file list
img_list = ["{:0>6d}.png".format(i) for i in range(0, 590+1, 5)]


# draw match. type=0 puts two images side by side; type=1 coincide two images
def draw_match(img1, img2, src_pts, ano_pts, type=0):
    src_pts = np.round(src_pts).astype(int)
    ano_pts = np.round(ano_pts).astype(int)
    if type == 0:
        h, w = img1.shape[:2]
        img_match = np.zeros((h, w*2), img1.dtype)
        img_match[0:h, 0:w] = img1
        img_match[0:h, w:w*2] = img2
        img_match = cv2.cvtColor(img_match, cv2.COLOR_GRAY2RGB)
        for i in range(len(src_pts)):
            col = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            p1 = (int(src_pts[i][0]), int(src_pts[i][1]))
            p2 = (int(ano_pts[i][0])+w, int(ano_pts[i][1]))
            p1l = (p1[0]-HALF_KERNEL_SIZE, p1[1]-HALF_KERNEL_SIZE)
            p1r = (p1[0]+HALF_KERNEL_SIZE, p1[1]+HALF_KERNEL_SIZE)
            p2l = (p2[0]-HALF_KERNEL_SIZE, p2[1]-HALF_KERNEL_SIZE)
            p2r = (p2[0]+HALF_KERNEL_SIZE, p2[1]+HALF_KERNEL_SIZE)
            img_match = cv2.line(img_match, p1, p2, color=col)
            img_match = cv2.rectangle(img_match, p1l, p1r, color=col)
            img_match = cv2.rectangle(img_match, p2l, p2r, color=col)

    else:
        img_match = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
        img_match = cv2.cvtColor(img_match, cv2.COLOR_GRAY2RGB)
        for i in range(len(src_pts)):
            img_match = cv2.line(img_match, tuple(src_pts[i]), tuple(ano_pts[i]), color=(255, 0, 0))
    
    return img_match


# find good matches. method can be "SIFT" or "SURF"; 
# threshold is used when creating feature point detector
def match(img1, img2, method="SIFT", threshold=10000):
    starttime = time.time()

    # compute feature points
    if method == "SURF":
        detector = cv2.xfeatures2d.SURF_create(threshold)
    else:
        version = cv2.__version__
        if version.split('.')[0]=="3":
            detector = cv2.xfeatures2d.SIFT_create(threshold)
        elif version.split('.')[0]=="4":
            detector = cv2.SIFT_create(threshold)

    kp1, descrip1 = detector.detectAndCompute(img1, None)
    kp2, descrip2 = detector.detectAndCompute(img2, None)

    # find good matches
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)

    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    match = flann.knnMatch(descrip1, descrip2, k=2)

    good=[]
    for i,(m,n) in enumerate(match):
        if(m.distance < 0.8*n.distance):
            good.append(m)

    print("\t# of good matches: %d" % len(good))

    # matched points on the first image
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,2)
    # corresponding matched points on the second image
    ano_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,2)

    # apply RANSAC to filter out imcorrect matches
    ransacReprojThreshold = 10.0
    M, mask = cv2.findHomography(src_pts, ano_pts, cv2.RANSAC, ransacReprojThreshold)
    matches_mask = mask.ravel().astype(np.bool)
    src_pts = src_pts[matches_mask]
    ano_pts = ano_pts[matches_mask]

    endtime = time.time()
    print("\tMatch time consume: %f" % (endtime - starttime))

    if SHOW_DEBUG_IMAGE:
        img_match = draw_match(img1, img2, src_pts, ano_pts, type=1)
        cv2.imshow("match", img_match)
        cv2.waitKey(0)
        cv2.destroyWindow("match")

    return src_pts, ano_pts


# whether the kernel covers some points out of image boundary
def kernel_coverage_inside(point, size):
    return point[0]-HALF_KERNEL_SIZE>=0 and point[0]+HALF_KERNEL_SIZE<size[1] \
        and point[1]-HALF_KERNEL_SIZE>=0 and point[1]+HALF_KERNEL_SIZE<size[0]


# compute averaged intensity difference around all matched points
def calc_diff(img1, img2, src_pts, ano_pts):
    starttime = time.time()

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE)) / (KERNEL_SIZE*KERNEL_SIZE)
    bias = 0
    count = 0

    for i in range(len(src_pts)):
        if kernel_coverage_inside(src_pts[i], img1.shape) and kernel_coverage_inside(ano_pts[i], img2.shape):
            src_area = img1[src_pts[i][1]-HALF_KERNEL_SIZE : src_pts[i][1]+HALF_KERNEL_SIZE+1, src_pts[i][0]-HALF_KERNEL_SIZE : src_pts[i][0]+HALF_KERNEL_SIZE+1]
            col1 = np.sum(src_area * kernel)    # average intensity around the i-th matched point in the first image
            ano_area = img2[ano_pts[i][1]-HALF_KERNEL_SIZE : ano_pts[i][1]+HALF_KERNEL_SIZE+1, ano_pts[i][0]-HALF_KERNEL_SIZE : ano_pts[i][0]+HALF_KERNEL_SIZE+1]
            col2 = np.sum(ano_area * kernel)    # average intensity around the i-th matched point in the second image

            bias += col2 - col1
            count += 1
    
    if count >= 5:
        # average over all matches
        bias /= count
    else:
        # insufficient # of matches!
        # in order to avoid excessive accidental error, let it contribute 0
        bias = 0
        print(">>> Too little valid matches! Skip this frame. <<<")

    print("\tBias: %f" % bias)

    endtime = time.time()
    print("\tCalc diff time consume: %f" % (endtime- starttime))

    if SHOW_DEBUG_IMAGE:
        img3 = np.round(np.clip(img2-bias, 0, 255)).astype(img2.dtype)
        cv2.imshow("img1", img1)
        cv2.imshow("img2", img2)
        cv2.imshow("img2 - bias", img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return bias


# usage: press p to pause, q to quit
if __name__ == "__main__":

    rospy.init_node('thermal_correction')
    IMAGE_PATH = rospy.get_param('~image_dir')
    OUTPUT_PATH = rospy.get_param('~output_dir')
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    # if the 1st image is too dark, enlarge this value
    # if it is too bright, put a negative value here
    START_GAIN = rospy.get_param('~start_gain')

    accumulate_diff = -START_GAIN
    diff_list = []

    ref_img = cv2.imread(os.path.join(IMAGE_PATH, img_list[0]), -1)
    print(ref_img.dtype)
    ref_img = np.clip(ref_img.astype(int)+START_GAIN, 0, 255).astype(ref_img.dtype)
    cv2.imwrite(os.path.join(OUTPUT_PATH, img_list[0]), ref_img)

    for i in range(1, len(img_list)):
        print("Processing image " + img_list[i])

        img1 = cv2.imread(os.path.join(IMAGE_PATH, img_list[i-1]), -1) if i==1 else img2
        img2 = cv2.imread(os.path.join(IMAGE_PATH, img_list[i]), -1)

        src_pts, ano_pts = match(img1, img2, "SIFT", 100)
        src_pts = np.round(src_pts).astype(int)
        ano_pts = np.round(ano_pts).astype(int)

        # diff is the intencity difference between this two images
        diff = calc_diff(img1, img2, src_pts, ano_pts)

        # accumulate diff give us the difference between this image and the first one (which is the reference)
        accumulate_diff += diff
        diff_list.append(accumulate_diff)

        print("accumulate diff: %f" % accumulate_diff)

        mask = img2>0
        img3 = img2.copy().astype(np.float64)
        # do the correction
        # img3[mask] -= accumulate_diff
        img3 -= accumulate_diff
        img3 = np.round(np.clip(img3, 0, 255)).astype(img2.dtype)
        cv2.imwrite(os.path.join(OUTPUT_PATH, img_list[i]), img3)

        if SHOW_RESULT_IMAGE:
            h, w = img1.shape[:2]
            result = np.zeros((h*2, w*2), dtype=img1.dtype)
            result[h:h*2, 0:w] = ref_img
            result[0:h, w:w*2] = img2
            result[h:h*3, w:w*2] = img3
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            result = cv2.putText(result, "reference", (10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            result = cv2.putText(result, "original", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            result = cv2.putText(result, "calibrated", (w+10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            result = cv2.resize(result, (0, 0), None, 0.9, 0.9, cv2.INTER_LINEAR)
            cv2.imshow("result", result)
            key = cv2.waitKey(100)
            if key == ord('p'):
                key = cv2.waitKey(0)
            if key == ord('q'):
                quit()

    print(diff_list)

    f_times = open(os.path.join(OUTPUT_PATH, "times.txt"), "w")
    for i in range(0, len(img_list)):
        f_times.write(img_list[i] + " " + str(i/2.0) + "\n")
    f_times.close()
