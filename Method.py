import cv2
import sys
import random
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
        """
        convert gray to binary (get the contours).
        """
        dst = gray.copy()
        dst[dst >= self.__threshold] = 255
        dst[dst < self.__threshold] = 0
        return dst

    def __GetMaxContourLoc(self, binary: np.ndarray) -> tuple:
        """
        evaluate the max contour and get the bounding box location.
        """
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

    def __CvtAreaLoc24Loc(self, rea_loc: tuple) -> np.ndarray:
        """
        convert the (x, y, w, h) to four potins including 
        top_left, top_right, bottom_left & bottom_right.
        """
        x, y, w, h = rea_loc
        top_left = [x, y]
        top_right = [x + w, y]
        bottom_left = [x, y + h]
        bottom_right = [x + w, y + h]

        return np.float32([top_left, top_right, bottom_left, bottom_right])

    def __ImageFiiltering(self, src: np.ndarray, reference: np.ndarray, blur_mode: str = "both") -> tuple:
        """
        src & reference do image blur to prevent image nosie affect alignment performance.
        """
        if self.__blur_ksize > 0:
            if blur_mode == "both":
                src = cv2.blur(src, self.__blur_mask)
                reference = cv2.blur(reference, self.__blur_mask)
            elif blur_mode == "src":
                src = cv2.blur(src, self.__blur_mask)
            elif blur_mode == "reference":
                reference = cv2.blur(reference, self.__blur_mask)
            else:
                print("Method.py(BlurOrbAlignImg): blur_mode " + blur_mode + " is not exist.")
        return (src, reference)

    def Calculate(self, src: np.ndarray, reference: np.ndarray, blur_mode: str = "both") -> tuple:
        """
        do alignment.
        """
        src_tmp = src.copy()
        ref_tmp = reference.copy()

        # Do bluring.
        src_tmp, ref_tmp = self.__ImageFiiltering(src_tmp, ref_tmp, blur_mode)
        
        # Convert images to grayscale
        src_gray = cv2.cvtColor(src_tmp, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(ref_tmp, cv2.COLOR_BGR2GRAY)

        # Convert gray to binary image.
        src_binary = self.__CvtGray2Binary(src_gray)
        ref_binary = self.__CvtGray2Binary(ref_gray)

        # Get max area location with x, y, w & h.
        src_max_loc = self.__GetMaxContourLoc(src_binary)
        ref_max_loc = self.__GetMaxContourLoc(ref_binary)

        # Convert (x, y, w, h) to four points.
        src_loc_4p = self.__CvtAreaLoc24Loc(src_max_loc)
        src_ref_4p = self.__CvtAreaLoc24Loc(ref_max_loc)


        # get transform matrix before perspective transform.
        transform_mat = cv2.getPerspectiveTransform(src_loc_4p, src_ref_4p)

        # Do alignment using perspective transform.
        dst_aligned = cv2.warpPerspective(src, transform_mat, 
                                          (reference.shape[1], reference.shape[0]))
        

        #print(src_max_loc)
        #print()
        #print(ref_max_loc)
        src_tmp_draw = src_tmp.copy()
        ref_tmp_draw = ref_tmp.copy()
        sx, sy, sw, sh = src_max_loc
        rx, ry, rw, rh = ref_max_loc
        cv2.rectangle(src_tmp_draw,(sx, sy),(sx + sw, sy + sh), (255,255,0),2)
        cv2.rectangle(ref_tmp_draw,(rx, ry),(rx + rw, ry + rh), (255,255,0),2)

        cv2.imshow("src_bbox_draw", src_tmp_draw)
        cv2.imshow("ref_bbox_draw", ref_tmp_draw)
        cv2.imshow("src_binary", src_binary)
        cv2.imshow("ref_binary", ref_binary)
        #cv2.imshow("dst_aligned", dst_aligned)
        return (dst_aligned, src_tmp_draw, ref_tmp_draw, src_binary, ref_binary)

 
def WriteAbsDiffImg(src1: np.ndarray, src2: np.ndarray, save_absimg_path: str) -> np.ndarray:
    """
    absolutely difference between src1 & src2 images, and save result.
    """
    dst_absdiff = cv2.absdiff(src1, src2)
    cv2.imwrite(save_absimg_path, dst_absdiff)

    return dst_absdiff
