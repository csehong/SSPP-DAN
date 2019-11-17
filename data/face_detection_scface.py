import cv2
import os
import copy
import glob


class FaceCrop(object):
    CASCADE_PATH_FRONT = "/Users/csehong/Google_Drive/Git_Repo/SSPP-DAN/data/haarcascades/haarcascade_frontalface_default.xml"
    # CASCADE_PATH_LEFT =
    # CASCADE_PATH_RIGHT =

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH_FRONT)

    def generate(self, img_input, min_size=10, output_size=224, show_result=False):

        img = copy.deepcopy(img_input)

        # Resize (224*? or ?*224)
        height, width = img.shape[:2]
        min_len = min(height, width)
        if min_len < 100:
            resize_ratio = 4*(output_size/float(min_len))
            img = cv2.resize(img, (int(resize_ratio*width), int(resize_ratio*height)))

        # Face Detection
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img_gray, 1.1, 3, minSize=(min_size, min_size))
        # Choose the Largest Region
        # print(faces, len(faces))
        if len (faces) == 0:
            return img_input
        else:
            face = faces[0, :]

        x, y, w, h = face
        r = max(w, h) / 2
        centerx = x + w / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)

        if (show_result):
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        img_face = img[ny:ny + nr, nx:nx + nr]
        img_face = cv2.resize(img_face, (output_size, output_size))

        return img_face


root_dir = '/Users/csehong/Google_Drive/Git_Repo/SSPP-DAN/data/SCface/mugshot_frontal_cropped_all'
output_str = 'faces_src'
output_dir = '/'.join(root_dir.split('/')[:-1] + [output_str] + [root_dir.split('/')[-1]])
if not(os.path.isdir(output_dir)):
    os.makedirs(os.path.join(output_dir))
output_dir_failed = output_dir + "_Failed"
if not(os.path.isdir(output_dir_failed)):
    os.makedirs(os.path.join(output_dir_failed))

cnt_failed = 0
face_detect = FaceCrop()
for root, dirs, files in os.walk(root_dir):

    for fname in files:
        img_path = os.path.join(root, fname)
        _, ext = os.path.splitext(img_path)

        is_mugshot = (fname.split('_')[1] == "frontal.JPG")
        if is_mugshot == False:
            cam_id = fname.split('_')[1][3]
            if int(cam_id) > 5:
                continue
        if ext not in ['.jpg', '.JPG', '.png', '.PNG']:
            continue

        # Load Image
        img = cv2.imread(img_path)

        # Face Detection and Save
        img_face = face_detect.generate(img)

        if img.size == img_face.size:
            print('********************** Detection Failed: %s**********************', fname)
            img_face_path = os.path.join(output_dir_failed, fname)
            cv2.imwrite("%s" % img_face_path, img)
            cnt_failed += 1
        else:
            img_face_path = os.path.join(output_dir, fname)
            cv2.imwrite("%s" % img_face_path, img_face)

print ('Fail Count: %d' %(cnt_failed))
