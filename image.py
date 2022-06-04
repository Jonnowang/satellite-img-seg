import numpy as np
from skimage import data, io, color, img_as_float

IMAGE_PATH = './images/test.png'

# Import training image and ground truth image into pytorch tensors
picture = io.imread(IMAGE_PATH)
picture = img_as_float(picture)
full_gt = io.imread('./images/gt.png', as_gray=True)

# Convert from numpy array into pytorch tensor and convert into 3 channel rgb
picture = np.split(picture, [3, 1], 2)
full_pic = picture[0]

# Store the number of rows and columns in variables
rows = full_pic.shape[0] // 256 + 1
columns = full_pic.shape[1] // 256 + 1

# Split the large image into a batch of smaller images of size 256 x 256 so that they fit in the specified gpu memory storage
pic_batch = np.zeros((rows * columns, 256, 256, 3))
gt_batch = np.zeros((rows * columns, 256, 256, 2))

# Split images into rows of 256 each
row_pic = np.array_split(full_pic, rows, 0)
row_gt = np.array_split(full_gt, rows, 0)

# Initialise counter
counter = 0

# Split each row into separate columns
for i in range(rows):
    pic_chunk = np.array_split(row_pic[i], columns, 1)
    gt_chunk = np.array_split(row_gt[i], columns, 1)
    # Store each individual image chunk into the picture batch
    for j in range(columns):
        image = pic_chunk[j]
        io.imsave("./image_chunks/pic_%i.png" % counter, image)
        gt_image = gt_chunk[j]
        io.imsave("./gt_chunks/gt_%i.png" % counter, gt_image)
        counter += 1