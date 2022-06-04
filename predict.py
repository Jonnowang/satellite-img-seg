import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io, color, img_as_float
from net import my_Net

IMAGE_PATH = './images/test.png'

# Load dimensions of test image
picture = io.imread(IMAGE_PATH)
rows = picture.shape[0] // 256 + 1
columns = picture.shape[1] // 256 + 1
chunk_width = round(picture.shape[0]/rows)
chunk_height = round(picture.shape[1]/columns)

# Number of processed images
image_count = next(os.walk('./image_chunks'))[2]
no_imgs = int(len(image_count))

pic_batch = list()

# Import test image into pytorch tensors
for i in range(no_imgs):
    pic_batch.append(io.imread('./image_chunks/pic_%i.png' % i))
    pic_batch[i] = img_as_float(pic_batch[i])

# Load trained network from file
net = my_Net()
net = net.double()
net.load_state_dict(torch.load('./learned.pth'))

imgs = list()

# Acquire input training image
for input in pic_batch:
    # Convert to pytorch tensor and permute to form of nSamples x nChannels x Height x Width
    pic = torch.from_numpy(input)
    pic = pic.permute(2, 0, 1)
    pic = torch.unsqueeze(pic, 0)
    pic = F.interpolate(pic, (chunk_width, chunk_height))

    # Pass pictures through trained neural network
    outputs = net(pic)
    outputs = outputs.permute(0, 2, 3, 1)
    outputs = torch.squeeze(outputs)
    output = outputs.detach().numpy()

    # Append outputs to a list
    imgs.append(output)

counter = 0
img_rows = list()

# Piece together image chunks to form rows
for i in range(rows):
    img_piece = np.hstack( (np.asarray(imgs[j + counter])) for j in range(columns) )
    counter += columns
    img_rows.append(img_piece)

# Piece together rows to form full image and save as prediction
img_comb = np.vstack(np.asarray(img_rows[k]) for k in range(rows))
prediction = np.zeros((img_comb.shape[0], img_comb.shape[1]))

# Convert 2 channel output to 1 channel grayscale image
for i in range(img_comb.shape[0]):
    for j in range(img_comb.shape[1]):
        prediction[i][j] = np.argmax(img_comb[i][j], axis=0)

# Save image as jpg
io.imsave("./predictions.jpg", prediction)