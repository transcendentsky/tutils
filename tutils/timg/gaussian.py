import numpy as np
import cv2

def prob_map_from_landmarks(ref_landmark, size=(384, 384), kernel_size=192):
    """
    Guassion Prob map from landmarks
    landmarks: [(x,y), (), ()....]
    size: (384,384)
    """
    landmarks = ref_landmark
    prob_maps = get_guassian_heatmaps_from_ref(landmarks=landmarks, num_classes=len(landmarks), \
                                                image_shape=size, kernel_size=kernel_size, sharpness=0.2)  #shape: (19, 800, 640)
    prob_map = np.sum(prob_maps, axis=0)
    prob_map = np.clip(prob_map, 0, 1)
    if False:
        print("====== Save Prob map to ./imgshow")
        cv2.imwrite(f"imgshow/prob_map_ks{kernel_size}.jpg", (prob_map*255).astype(np.uint8) )
    return prob_map

def select_point_from_prob_map(prob_map, size=(192, 192)):
    size_x, size_y = prob_map.shape
    assert size_x == size[0]
    assert size_y == size[1]
    chosen_x1 = np.random.randint(int(0.1 * size_x), int(0.9 * size_x))
    chosen_x2 = np.random.randint(int(0.1 * size_x), int(0.9 * size_x))
    chosen_y1 = np.random.randint(int(0.1 * size_y), int(0.9 * size_y))
    chosen_y2 = np.random.randint(int(0.1 * size_y), int(0.9 * size_y))
    if prob_map[chosen_x1, chosen_y1] * np.random.random() > prob_map[chosen_x2, chosen_y2] * np.random.random() :
        return chosen_x1, chosen_y1
    else:
        return chosen_x2, chosen_y2
    

def get_guassian_heatmaps_from_ref(landmarks, num_classes, image_shape=(800, 640), kernel_size=96, sharpness=0.2):
    """
    input: landmarks, [(1,1), ...]
    num_classes: number of classes
    image_shape: shape of the original image
    return:  shape: (19, 800, 640)
    """
    assert len(landmarks) == num_classes
    heatmaps = []
    for landmark in landmarks:
        # print("landmark: ", landmark)
        heatmaps.append(get_gaussian_heatmap_from_point(landmark, image_shape, size=kernel_size, sharpness=sharpness))
    return np.stack(heatmaps, axis=0)  # shape: (19, 800, 640)


def get_gaussian_heatmap_from_point(center_location, heatmap_shape=(96, 96), size=8, sharpness=0.2):
    """
    center_location: [x, y] the location the center of normal distribution
    heatmap_shape: the shape of output map, e.g. (96,96)
    size: size of heat area
    """
    # center_location[0], center_location[1] = int(center_location[0]), int(center_location[1])
    # assert type(center_location[0]) == int, "Expected int, bug got {}".format(type(center_location[0]))
    # assert type(center_location) == np.ndarray
    _, _, z = build_gaussian_layer(0, 1, len=size, sharpness=sharpness)
    # print("z.shape", z.shape)
    z = z - np.min(z)
    z_max = np.max(z)
    z /= z_max
    location_left = int(center_location[0])
    location_right = int(center_location[0] + size*2-1)
    location_top = int(center_location[1])
    location_bottom = int(center_location[1] + size*2-1)
    heatmap = np.zeros((heatmap_shape[0]+size*2, heatmap_shape[1]+size*2))
    # print(location_left, location_right)
    # import ipdb;ipdb.set_trace()
    heatmap[location_top:location_bottom, location_left:location_right] = z
    final_map = heatmap[size:-size, size:-size]
    assert final_map.shape[0] == heatmap_shape[0]
    return final_map

def build_gaussian_layer(mean, standard_deviation, step=1, len=8, sharpness=0.2):
    """
    copy from blog.csdn.net
    """
    # scaled_size = int(len / sharpness)
    scaled_size = len
    center_point = (scaled_size, scaled_size)
    x = np.arange(-scaled_size + 1, scaled_size, step) * (sharpness**2)
    y = np.arange(-scaled_size + 1, scaled_size, step) * (sharpness**2)
    x, y = np.meshgrid(x, y)
    z = np.exp(-((y - mean) ** 2 + (x - mean) ** 2) / (2 * (standard_deviation ** 2)))
    z = z / (np.sqrt(2 * np.pi) * standard_deviation)
    return (x, y, z)

def test_get_guassian_heatmaps_from_ref():
    a = get_guassian_heatmaps_from_ref([(100,100), (101,101)], num_classes=2, image_shape=(192, 192), kernel_size=96)
    a = a - np.min(a)
    a = a / np.max(a) * 255
    index = np.where(a[0]>0, 255, 0)
    print("a index", np.sum(index))
    aaa = a
    print(a[0].shape, np.max(a), np.min(a))
    print(a)
    import ipdb; ipdb.set_trace()
    cv2.imwrite("imgshow/gp-1.jpg", a[0].astype(np.uint8))
    cv2.imwrite("imgshow/gp-index-1.jpg", index.astype(np.uint8))

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # fig = plt.figure()
    # ax = Axes3D(fig)
    #
    # x3, y3, z3 = build_gaussian_layer(0, 1)
    # print("shapes: ", x3.shape)
    # import ipdb;
    # ipdb.set_trace()
    # ax.plot_surface(x3, y3, z3, rstride=1, cstride=1, cmap='rainbow')
    # plt.show()
    # plt.savefig("test_guassion.png")
    test_get_guassian_heatmaps_from_ref()