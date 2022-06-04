import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io, color, img_as_float
from net import my_Net

# Train network on GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load dimensions of training image
picture = io.imread('./images/rgb.png')
rows = picture.shape[0] // 256 + 1
columns = picture.shape[1] // 256 + 1
chunk_width = round(picture.shape[0]/rows)
chunk_height = round(picture.shape[1]/columns)

# Number of processed images
image_count = next(os.walk('./image_chunks'))[2]
no_imgs = int(len(image_count))

# Empty list
pic_batch = list()
gt_temp = list()
gt_batch = list()
dataset = list()

# Import training image and ground truth image into pytorch tensors
for i in range(no_imgs):
    pic_batch.append(io.imread('./image_chunks/pic_%i.png' % i))
    pic_batch[i] = img_as_float(pic_batch[i])
    gt_temp.append(io.imread('./gt_chunks/gt_%i.png' % i, as_gray=True))

# Set each pixel in the ground truth to either black or white in each channel
for gt in gt_temp:
    # Create empty numpy array
    gt_new = np.zeros((gt.shape[0], gt.shape[1], 2))
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            # If black
            if gt[i][j] == 0:
                gt_new[i][j][0] = 1
                gt_new[i][j][1] = 0
            # If white
            elif gt[i][j] == 255:
                gt_new[i][j][0] = 0 
                gt_new[i][j][1] = 1
            else:
                print("Error: input image not grayscale")
    gt_batch.append(gt_new)

# Convert images and gts to nSamples x nChannels x Height x Width before passing into network for processing
for j in range(len(pic_batch)):
    pic = torch.from_numpy(pic_batch[j])
    pic = pic.permute(2, 0, 1)
    pic = torch.unsqueeze(pic, 0)
    pic = F.interpolate(pic, (chunk_width, chunk_height))

    gt = torch.from_numpy(gt_batch[j])
    gt = gt.permute(2, 0, 1)
    gt = torch.unsqueeze(gt, 0)
    gt = F.interpolate(gt, (chunk_width, chunk_height))

    dataset.append((pic, gt))

# Initialise network and place on the GPU
net = my_Net()
net = net.double()
net = net.to(device)
losses = list()

# Specify loss and optimiser criterion
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=0.01)

"""SET STOP THRESHOLD"""
stop_threshold = 0.15

# Iterate epochs until interrupted by keyboard command
try:
    while True:
        running_loss = 0
        for data in dataset:
            # get the inputs from the data
            inputs, truths = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forwards
            outputs = net(inputs)
            # backwards
            loss = criterion(outputs, truths)   
            loss.backward()
            # optimise
            optimizer.step()

            # Add the loss for each mini batch to the running loss
            running_loss += loss.item()
        losses.append(running_loss / len(dataset))
        print(running_loss / len(dataset))
        # Terminate learning when threshold of desired training loss is reached
        if running_loss / len(dataset) < stop_threshold:
            raise KeyboardInterrupt
except KeyboardInterrupt:
    # Plot training loss as a function of epoch
    plt.plot(losses)
    plt.xlabel("Epoch (Number of Iterations)")
    plt.ylabel("Training Loss")
    plt.savefig("./loss_vs_epoch.png")
    plt.show()

    # Save learned network
    torch.save(net.state_dict(), './learned.pth')