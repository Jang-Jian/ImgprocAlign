import cv2

import Method


if __name__ == "__main__":
    image_path = "./images/Cameraman.png"
    save_fmatchs_img_path = "Lenna_shiftX_10_shifty_10_noiseP_5_feature_matchs.jpg"
    save_difference_map_path = "Lenna_absdiff_alignedim2_im1.jpg"
    shift_x, shift_y = -10, -10
    noise_percent = 0.09
    max_features = 500
    good_match_percent = 0.2
    src_blur_ksize = 5


    # 1. Take image IM1.
    im1 = cv2.imread(image_path)

    # 2. Shift the image X,Y pixels, that is imIM2 (shift X & Y more than 10 pixels).
    imIM2 = Method.CenterShiftImg(im1, shift_x, shift_y)

    # 3. Add random noise to IM2.
    im2 = Method.AddNoise2Img(imIM2, noise_percent)

    # 4. Develop in python alignment method between the 2 images (p.s. im2 align to im1).
    # p.s. Blur the img2 noise image, which prevent the noise effect the effecient point for orb feature point detection.
    # alignment template (reference): im1.
    # aligned image (src): im2.
    im2_aligned = Method.BlurOrbAlignImg(im2, im1, max_features, good_match_percent, 
                                         src_blur_ksize, save_fmatchs_img_path)

    # Output difference map.
    difference_map = Method.WriteAbsDiffImg(im1, im2_aligned, save_difference_map_path)


    cv2.imshow("image", im1)
    cv2.imshow("shift_image", imIM2)
    cv2.imshow("nosie_image", im2)
    cv2.imshow("img2_align_im1", im2_aligned)
    cv2.imshow("abs(After shift image im2 - im1)", difference_map)
    #cv2.imshow("absdiff_im2", absdiff_im2)
    cv2.waitKey(0)