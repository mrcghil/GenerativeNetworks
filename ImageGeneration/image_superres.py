import cv2 
from cv2 import dnn_superres

# Create an SR object - only function that differs from c++ code
sr = dnn_superres.DnnSuperResImpl_create()

# Read image
image = cv2.imread('C:\WORKSPACES\ZINKY\GenerativeNetworks\InputFiles\ImageCollection\giraffe.jpg')

# Read the desired model
path = "C:\\WORKSPACES\\ZINKY\\GenerativeNetworks\\SuperRes Models\\EDSR_x4.pb"
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 4)
# Upscale the image
result = sr.upsample(image)
# Save the image
cv2.imwrite("C:\WORKSPACES\ZINKY\GenerativeNetworks\Results\HighRes\giraffe.jpg", result)