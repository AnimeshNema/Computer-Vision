import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim

# Instantiating my model named Net, saved in model.py
from model import Net
net = Net()
print(net)

# from my data_transformation.py file.
from data_transformation import FacialKeypointsDataset
from data_transformation import Rescale, RandomCrop, Normalize, ToTensor

data_transform = transforms.Compose([Rescale(250), RandomCrop(224), Normalize(), ToTensor()])

# create the transformed dataset
transformed_dataset = FacialKeypointsDataset(csv_file='/data/training_frames_keypoints.csv',
                                             root_dir='/data/training/',
                                             transform=data_transform)

print('Number of images: ', len(transformed_dataset))

# iterate through the transformed dataset and print some stats about the first few samples
# for i in range(5):
#     plt.figure(figsize=(20,10))
#     plt.subplot(1,5,i+1)
#     sample = transformed_dataset[i]
#     image = sample['image'].numpy()
#     keypoints = sample['keypoints']
#     #keypoints = keypoints.view(keypoints.size()[0], 68, -1)
#     keypoints = keypoints.data.numpy()
#     keypoints = keypoints*50.0+100
#     plt.imshow(np.squeeze(image))
#     print('keypoints are:', keypoints)
#     plt.scatter(keypoints[:,0], keypoints[:,1], marker='.')
#     print(i, sample['image'].size(), sample['keypoints'].size())
#     plt.show()

# load training data in batches
batch_size = 30
train_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# create the test dataset
test_dataset = FacialKeypointsDataset(csv_file='/data/test_frames_keypoints.csv', root_dir='/data/test/', transform=data_transform)

# load test data in batches
batch_size = 30
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# test the model on a batch of test images

# before traning, lets test performance.
def net_sample_output():

    # iterate through the test dataset
    for i, sample in enumerate(test_loader):

        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']

        # convert images to FloatTensors
        images = images.type(torch.FloatTensor)

        # forward pass to get net output
        output_pts = net(images)

        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)

        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts

# test_images, test_outputs, gt_pts = net_sample_output()

# # print out the dimensions of the data to see if they make sense
# print(test_images.data.size())
# print(test_outputs.data.size())
# print(gt_pts.size())

#Visualize the output
def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=5):

    for i in range(batch_size):
        plt.figure(figsize=(20,10))
        ax = plt.subplot(1, batch_size, i+1)

        # un-transform the image data
        image = test_images[i].data   # get the image from it's Variable wrapper
        image = image.numpy()   # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.numpy()
        # undo normalization of keypoints
        predicted_key_pts = predicted_key_pts*50.0+100

        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]
            ground_truth_pts = ground_truth_pts*50.0+100

        # call show_all_keypoints
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)

        #plt.axis('off')

    plt.show()

# Training parameters
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(net.parameters(), lr = 0.003)

def train_net(n_epochs):

    # prepare the net for training
    net.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
                running_loss = 0.0

    print('Finished Training')


def eval_net(n_epochs):

    # prepare the net for training
    net.eval()

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(test_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # print loss statistics
            running_loss += loss.item()
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
                running_loss = 0.0

    print('Finished Evaluation')

n_epochs = 10
train_net(n_epochs)

#Evaluate net
eval_net(n_epochs)

# get a sample of test data
test_images, test_outputs, gt_pts = net_sample_output()

#Visualize the output
Viz = visualize_output(test_images, test_outputs, gt_pts)

model_dir = 'saved_models/'
model_name = 'mark1_face.pt'

# after training, save your model parameters in the dir 'saved_models'
torch.save(net.state_dict(), model_dir+model_name)
