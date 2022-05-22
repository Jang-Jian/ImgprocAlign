import cv2
import sys
import random
from cv2 import threshold
import numpy as np


def CenterShiftImg(src: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    """
    shift the image.
    """
    print("shift_X:", shift_x, ", shift_Y:", shift_y)
    affine_mat = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    dst = cv2.warpAffine(src, affine_mat, (src.shape[1], src.shape[0]))
    return dst


def AddNoise2Img(src: np.ndarray, noise_percent: float = 0.0) -> np.ndarray:
    """
    create pepper(gray level=0) and salt(gray level=255).
    """
    height, width = src.shape[0], src.shape[1]
    rand_num = int(height * width * noise_percent)
    dst = src.copy()

    for index in range(0, rand_num):
        row_index = random.randint(0, height - 1)
        col_index = random.randint(0, width - 1)
        nosie_type = random.randint(0, 1)
        #print(nosie_type)
        dst[row_index, col_index] = 255 * nosie_type
    return dst


class Alignment2D(object):
    """
    2-dimension image alignment.
    """
    def __init__(self, blur_ksize: int = 5, threshold: int = 50) -> None:
        self.__blur_ksize = blur_ksize
        self.__blur_mask = (self.__blur_ksize, self.__blur_ksize)
        self.__threshold = threshold

    def __CvtGray2Binary(self, gray: np.ndarray) -> np.ndarray:
        dst = gray.copy()
        dst[dst >= self.__threshold] = 255
        dst[dst < self.__threshold] = 0
        return dst

    def __GetMaxContourLoc(self, binary: np.ndarray) -> tuple:
        _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area = 0
        max_area_loc = None
        # find the biggest bounding box.
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = (x + w) * (y + h)
            if area > max_area:
                max_area = area
                max_area_loc = (x, y, w, h)
        
        return max_area_loc

    def Calculate(self, src: np.ndarray, reference: np.ndarray, blur_mode: str = "both"):
        src_tmp = src.copy()
        ref_tmp = reference.copy()

        # Do bluring.
        if self.__blur_ksize > 0:
            if blur_mode == "both":
                src_tmp = cv2.blur(src_tmp, self.__blur_mask)
                ref_tmp = cv2.blur(ref_tmp, self.__blur_mask)
            elif blur_mode == "src":
                src_tmp = cv2.blur(src_tmp, self.__blur_mask)
            elif blur_mode == "reference":
                ref_tmp = cv2.blur(ref_tmp, self.__blur_mask)
            else:
                print("Method.py(BlurOrbAlignImg): blur_mode " + blur_mode + " is not exist.")
        
        # Convert images to grayscale
        src_gray = cv2.cvtColor(src_tmp, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(ref_tmp, cv2.COLOR_BGR2GRAY)

        # Convert gray to binary image.
        src_binary = self.__CvtGray2Binary(src_gray)
        ref_binary = self.__CvtGray2Binary(ref_gray)

        # Get max area location with x, y, w & h.
        src_max_loc = self.__GetMaxContourLoc(src_binary)
        ref_max_loc = self.__GetMaxContourLoc(ref_binary)
        
        src_tmp_draw = src_tmp.copy()
        ref_tmp_draw = ref_tmp.copy()
        sx, sy, sw, sh = src_max_loc
        rx, ry, rw, rh = ref_max_loc
        cv2.rectangle(src_tmp_draw,(sx, sy),(sx + sw, sy + sh), (255,255,0),2)
        cv2.rectangle(ref_tmp_draw,(rx, ry),(rx + rw, ry + rh), (255,255,0),2)

        cv2.imshow("src_tmp_draw", src_tmp_draw)
        cv2.imshow("ref_tmp_draw", ref_tmp_draw)

        # use top-left, top-right, bottom-left & bottom-right 4 points to do alignment. 
    


def BlurOrbAlignImg(src: np.ndarray, reference: np.ndarray, 
                    max_features: int = 500, good_match_percent: float = 0.15, 
                    src_blur_ksize: int = 5, blur_mode: str = "both", save_fmatchs_img_path: str = ""):
    """
    align the src image by reference image by Oriented FAST method.
    ORB: An efficient alternative to SIFT or SURF.
    """
    src_tmp = src.copy()
    ref_tmp = reference.copy()
    if src_blur_ksize > 0:
        if blur_mode == "both":
            src_tmp = cv2.blur(src_tmp, (src_blur_ksize, src_blur_ksize))
            ref_tmp = cv2.blur(ref_tmp, (src_blur_ksize, src_blur_ksize))
        elif blur_mode == "src":
            src_tmp = cv2.blur(src_tmp, (src_blur_ksize, src_blur_ksize))
        elif blur_mode == "reference":
            ref_tmp = cv2.blur(ref_tmp, (src_blur_ksize, src_blur_ksize))
        else:
            print("Method.py(BlurOrbAlignImg): blur_mode " + blur_mode + " is not exist.")

    # Convert images to grayscale
    src_gray = cv2.cvtColor(src_tmp, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_tmp, cv2.COLOR_BGR2GRAY)
    
    


    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(src_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(ref_gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * good_match_percent)
    matches = matches[:numGoodMatches]

    # Draw top matches
    if save_fmatchs_img_path != "":
        img_matches = cv2.drawMatches(src_tmp, keypoints1, reference, keypoints2, matches, None)
        cv2.imwrite(save_fmatchs_img_path, img_matches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find transform matrix using homography with RANSAC.
    # RANSAC: RANdom SAmple Consensus
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC, confidence=0.9)

    # Use homography with 
    height, width, channels = reference.shape
    dst_aligned = cv2.warpPerspective(src, h, (width, height))

    return dst_aligned


def WriteAbsDiffImg(src1: np.ndarray, src2: np.ndarray, save_absimg_path: str) -> np.ndarray:
    """
    absolutely difference between src1 & src2 images, and save result.
    """
    dst_absdiff = cv2.absdiff(src1, src2)
    cv2.imwrite(save_absimg_path, dst_absdiff)

    return dst_absdiff