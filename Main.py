import os
import cv2
import Method


if __name__ == "__main__":
    image_path = "./images/Airplane.png"
    
    save_match_img_root_path = "./"
    shift_x, shift_y = 20, 20
    noise_percent = 0.1
    blur_ksize = 7
    gray_threshold = 50

    ############################################################
    print("image_path:", image_path)
    print("noise_percent:", str(noise_percent * 100) + "%")
    print("blur_ksize:", blur_ksize)
    print("gray_threshold:", gray_threshold)

    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    save_org_im2_path = os.path.join(save_match_img_root_path, filename + "_spNosieP_" + \
                                    str(noise_percent) + "_img2.png")
    save_im1_alignment_path = os.path.join(save_match_img_root_path, filename + "_spNosieP_" + str(noise_percent) + 
                                 "_grayThreshold_" + str(gray_threshold) + \
                               "_blurKsize_" + str(blur_ksize) + "_im1_align2_img2.jpg")
    save_im2_alignment_path = os.path.join(save_match_img_root_path, filename + "_spNosieP_" + str(noise_percent) + 
                                 "_grayThreshold_" + str(gray_threshold) + \
                               "_blurKsize_" + str(blur_ksize) + "_im2_align2_img1.jpg")

    save_src_draw_bbox_path = os.path.join(save_match_img_root_path, filename + "_spNosieP_" + str(noise_percent) + 
                                 "_grayThreshold_" + str(gray_threshold) + \
                               "_blurKsize_" + str(blur_ksize) + "_src_draw_bbox.jpg")
    
    save_ref_draw_bbox_path = os.path.join(save_match_img_root_path, filename + "_spNosieP_" + str(noise_percent) + 
                                 "_grayThreshold_" + str(gray_threshold) + \
                               "_blurKsize_" + str(blur_ksize) + "_ref_draw_bbox.jpg")

    save_src_binary_path = os.path.join(save_match_img_root_path, filename + "_spNosieP_" + str(noise_percent) + 
                                 "_grayThreshold_" + str(gray_threshold) + \
                               "_blurKsize_" + str(blur_ksize) + "_src_binary.jpg")
    
    save_ref_binary_path = os.path.join(save_match_img_root_path, filename + "_spNosieP_" + str(noise_percent) + 
                                 "_grayThreshold_" + str(gray_threshold) + \
                               "_blurKsize_" + str(blur_ksize) + "_ref_binary.jpg")

    save_difference_map_path = os.path.join(save_match_img_root_path, filename + "_spNosieP_" + str(noise_percent) + 
                                 "_grayThreshold_" + str(gray_threshold) + \
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
    im1_align2_im2 = Method.Alignment2D(blur_ksize, gray_threshold)
    im1_aligned_results = im1_align2_im2.Calculate(im1, im2, "both")
    im1_aligned, src_tmp_draw, ref_tmp_draw, src_binary, ref_binary = im1_aligned_results

    # im2 align to im1.
    # alignment template (reference): im1.
    # aligned image (src): im2.
    im2_align2_imd = Method.Alignment2D(blur_ksize, gray_threshold)
    im2_aligned_results = im2_align2_imd.Calculate(im2, im1, "both")
    im2_aligned, src_tmp_draw, ref_tmp_draw, src_binary, ref_binary = im2_aligned_results

    

    # Output difference map.
    difference_map = Method.WriteAbsDiffImg(im1_aligned, im2_aligned, save_difference_map_path)


    # save experimential results.
    cv2.imwrite(save_org_im2_path, im2)
    cv2.imwrite(save_im1_alignment_path, im1_aligned)
    cv2.imwrite(save_im2_alignment_path, im2_aligned)
    cv2.imwrite(save_difference_map_path, difference_map)
    cv2.imwrite(save_src_draw_bbox_path, src_tmp_draw)
    cv2.imwrite(save_ref_draw_bbox_path, ref_tmp_draw)
    cv2.imwrite(save_src_binary_path, src_binary)
    cv2.imwrite(save_ref_binary_path, ref_binary)



    cv2.imshow("im1", im1)
    cv2.imshow("im2 (noise percent: " + str(noise_percent * 100) + "%)", im2)
    cv2.imshow("im1_aligned_to_im2 (result1)", im1_aligned)
    cv2.imshow("im2_aligned_to_im1 (result2)", im2_aligned)
    cv2.imshow("difference_map  abs(result1 - result2)", difference_map)
    #cv2.imshow("absdiff_im2", absdiff_im2)
    cv2.waitKey(0)