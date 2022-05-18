import cv2
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


def BlurOrbAlignImg(src: np.ndarray, reference: np.ndarray, 
                    max_features: int = 500, good_match_percent: float = 0.15, 
                    src_blur_ksize: int = 5, save_fmatchs_img_path: str = ""):
    """
    align the src image by reference image by Oriented FAST method.
    ORB: An efficient alternative to SIFT or SURF.
    """
    src_tmp = src.copy()
    ref_tmp = reference.copy()
    if src_blur_ksize > 0:
        src_tmp = cv2.blur(src_tmp, (src_blur_ksize, src_blur_ksize))
        ref_tmp = cv2.blur(ref_tmp, (src_blur_ksize, src_blur_ksize))

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

    # Find homography with RANSAC.
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