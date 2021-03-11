# Augment
`timg`
- compress_JPEG_from_path(path, quality=10)
    - JPEG image compressing
- gaussian_blur(img, kernel=(3, 3))
- class Augment
- partial_augment(img, augment)


# Printing
`print_img`
- print_img_auto(img, img_type='ori', is_gt=True, fname=None)
    - type="img"/"exr"/"bg" ...
    - save_img by cv2

`draw_heatmap`
- draw_heatmap(points: np.ndarray, points2, fname="testtt.png")
- draw_scatter(points, points2, fname="ttest.png", c="red")

