import torch
from alexnet_model import AlexNet
from vgg_model import vgg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# create model
model = vgg(model_name="vgg16", num_classes=5)
# model = resnet34(num_classes=5)
# load model weights
model_weight_path = "./vgg16Net.pth"  # "./resNet34.pth"
model.load_state_dict(torch.load(model_weight_path))
print(model)

# load image
img = Image.open("./tulip.jpeg")
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# forward
out_put = model(img)
for feature_map in out_put:
    # [N, C, H, W] -> [C, H, W]
    im = np.squeeze(feature_map.detach().numpy())
    # [C, H, W] -> [H, W, C]
    im = np.transpose(im, [1, 2, 0])

    # show top 12 feature maps
    plt.figure()
    for i in range(12):
        ax = plt.subplot(3, 4, i+1)
        # [H, W, C]
        plt.imshow(im[:, :, i])
    plt.show()

