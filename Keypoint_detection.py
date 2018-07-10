import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from models import Net

import cv2
# load in color image for face detection
image = cv2.imread('images/obamas.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plot the image
fig = plt.figure(figsize=(9,9))
plt.imshow(image)
#plt.show()

# load in a haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(image, 1.2, 2)

# make a copy of the original image to plot detections on
image_with_detections = image.copy()

# loop over the detected faces, mark the image where each face is found
for (x,y,w,h) in faces:
    # draw a rectangle around each detected face
    cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3)

fig = plt.figure(figsize=(9,9))

plt.imshow(image_with_detections)
#plt.show()


# Load model
net = Net()
net.load_state_dict(torch.load('saved_models/mark1_face.pt'))

# test model
net.eval()

image_copy = np.copy(image)
from torch.autograd import Variable
# loop over the detected faces from haar cascade
for (x,y,w,h) in faces:
    # Select the region of interest that is the face in the image
    roi = image_copy[y:y+h, x:x+w]
    #print(roi.shape)
    # Convert the face region from RGB to grayscale
    roi_gray= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #print(roi_gray.shape)
    # Normalize the grayscale image
    roi_norm = roi_gray/255
    # Rescale the detected face to be the expected square size for your CNN
    roi_rescale = cv2.resize(roi_norm,(224,224))
    roi_rescale = roi_rescale.reshape(roi_rescale.shape[0], roi_rescale.shape[1], 1)
    #print(roi_rescale.shape)
    # Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
    roi_tensor = roi_rescale.transpose((2,0,1))
    roi_tensor = torch.from_numpy(roi_tensor)
    #print(roi_tensor.shape)
    # Make facial keypoint predictions using loaded, trained network
    roi_tensor= roi_tensor.unsqueeze_(0)
    roi_tensor = roi_tensor.type(torch.FloatTensor)
    roi_tensor = Variable(roi_tensor)
    output_pts = net(roi_tensor)
    # Display each detected face and the corresponding keypoints
    output_pts = output_pts.view(output_pts.size()[0], 68, -1)
    output_pts = output_pts.data.numpy()
    #output_pts = output_pts.numpy()
    output_pts = output_pts*75.0+100
    #print(output_pts.shape)

    fig = plt.figure()
    fig.add_subplot(1,len(faces),1)
    plt.imshow(np.squeeze(roi_rescale), cmap='gray')
    plt.scatter(output_pts[:,:,0],output_pts[:,:,1] ,s=20, marker='.', c='m')
    plt.show()
