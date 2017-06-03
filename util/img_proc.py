import cv2, random



def resize(img, size=(224, 224)):
    return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)


def crop_corners(img, crop_size=(224, 224)):
    (x, y, _) = img.shape
    (cx, cy) = crop_size
    four_crops = [[0, 0], [x - cx, 0], [0, y - cy], [x - cx, y - cy]]
    crop_imgs = []
    for rect in four_crops:
        crop_imgs.append(img[rect[0]:rect[0] + cx, rect[1]:rect[1] + cy])
    return crop_imgs


def crop_corner(img, p, crop_size=(224, 224)):
    (x, y, _) = img.shape
    (cx, cy) = crop_size

    if p == 0:
        return crop_center(img, crop_size)

    four_crops = [[0, 0], [x - cx, 0], [0, y - cy], [x - cx, y - cy]]
    rect = four_crops[p-1]
    crop_img = img[rect[0]:rect[0] + cx, rect[1]:rect[1] + cy]

    return crop_img


def crop_center(img, crop_size=(224, 224)):
    (x, y, _) = img.shape
    (cx, cy) = crop_size
    return img[(x - cx) // 2:(x - cx) // 2 + cx, (y - cy) // 2:(y - cy) // 2 + cy]


def crop_corners_center(img, crop_size=(224, 224)):
    return crop_corners(img, crop_size) + [crop_center(img, crop_size)]


def rand_flip(img):
    if random.randint(0, 1) == 0:
        return img[:, ::-1]
    else:
        return img


def flip(img):
    return img[:, ::-1]
