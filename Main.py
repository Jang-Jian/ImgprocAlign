import os
import cv2
import Method


if __name__ == "__main__":
    image_path = "./images/Lenna.png"
    
    
    save_match_img_root_path = "./"
    shift_x, shift_y = 20, 20
    noise_percent = 0.05
    max_features = 500
    good_match_percent = 0.2
    blur_ksize = 7
    threshold = 50

    ############################################################
    print("image_path:", image_path)
    print("noise_percent:", str(noise_percent * 100) + "%")
    print("blur_ksize:", blur_ksize)

    filename = os.path.splitext(os.path.basename(image_path))[0]
    save_m1_match_m2_img_path = os.path.join(save_match_img_root_path, filename + "_im1Matchim2_shiftX_" + str(shift_x) + "_shiftY_" + str(shift_y) + \
                                "_spNosieP_" + str(noise_percent) + "_maxFeatures_" + str(max_features) + "_matchP_" + str(good_match_percent) + \
                                "_blurKsize_" + str(blur_ksize) + ".jpg")
    save_m2_match_m1_img_path = os.path.join(save_match_img_root_path, filename + "_im2Matchim1_shiftX_" + str(shift_x) + "_shiftY_" + str(shift_y) + \
                                "_spNosieP_" + str(noise_percent) + "_maxFeatures_" + str(max_features) + "_matchP_" + str(good_match_percent) + \
                                "_blurKsize_" + str(blur_ksize) + ".jpg")
    save_difference_map_path = os.path.join(save_match_img_root_path, filename + "_spNosieP_" + str(noise_percent) + \
                               "_blurKsize_" + str(blur_ksize) + "_absdiff_after_shift.jpg")
    

    # 1. Take image IM1.
    im1 = cv2.imread(image_path)

    # 2. Shift the image X,Y pixels, that is imIM2 (shift X & Y more than 10 pixels).
    imIM2 = Method.CenterShiftImg(im1, shift_x, shift_y)

    # 3. Add random noise to IM2.
    im2 = Method.AddNoise2Img(imIM2, noise_percent)

    # 4. Develop in python alignment method between the 2 images.
    # p.s. Blur im1 & im2 noise image, which prevent the noise effect the effecient point for orb feature point detection.
    
    # im1 align to im2.
    # alignment template (reference): im2.
    # aligned image (src): im1.
    im1_aligned = Method.BlurOrbAlignImg(im1, im2, max_features, good_match_percent, 
                                         blur_ksize, "both", save_m1_match_m2_img_path)

    # to do ...
    im_aligned_imd = Method.Alignment2D(blur_ksize, threshold)
    im_aligned_imd.Calculate(im1, im2, "both")

    # im2 align to im1.
    # alignment template (reference): im1.
    # aligned image (src): im2.
    im2_aligned = Method.BlurOrbAlignImg(im2, im1, max_features, good_match_percent, 
                                         blur_ksize, "both", save_m2_match_m1_img_path)

    

    # Output difference map.
    difference_map = Method.WriteAbsDiffImg(im1_aligned, im2_aligned, save_difference_map_path)


    cv2.imshow("im1", im1)
    cv2.imshow("im2 (noise percent: " + str(noise_percent * 100) + "%)", im2)
    cv2.imshow("im1_aligned_to_im2 (result1)", im1_aligned)
    cv2.imshow("im2_aligned_to_im1 (result2)", im2_aligned)
    cv2.imshow("difference_map  abs(result1 - result2)", difference_map)
    #cv2.imshow("absdiff_im2", absdiff_im2)
    cv2.waitKey(0)