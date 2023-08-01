import ssl
import urllib

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

ssl._create_default_https_context = ssl._create_unverified_context


class EasyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # in channels, out channels, kernel_size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# timm model path default is .cache/torch/hub/checkpoints/xxx.pth
class InceptionV4():
    def __init__(self) -> None:
        self.model_name = 'inception_v4'
        self.model = timm.create_model(self.model_name, pretrained=True)
        self.model.eval()

    def preprocess(self):
        self.config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**self.config)

    def inference(self):
        url, filename = ('https://github.com/pytorch/hub/raw/master/images/dog.jpg', 'dog.jpg')
        urllib.request.urlretrieve(url, filename)
        img = Image.open(filename).convert('RGB')  # PIL default is 4 channel, additional one is alpha
        tensor = self.transform(img).unsqueeze(0)  # add a dimension (denote batch for model)
        with torch.no_grad():
            out = self.model(tensor)
        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        print(probabilities.shape, torch.argmax(probabilities), probabilities[torch.argmax(probabilities)])

        # get the classes from url
        url, filename = ('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt', 'imagenet_classes.txt')
        urllib.request.urlretrieve(url, filename)
        with open('imagenet_classes.txt', 'r') as f:
            categories = [s.strip() for s in f.readlines()]

        # Print top categories per image
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        for i in range(top5_prob.size(0)):
            print(f'top {i+1} is {categories[top5_catid[i]]}, with confidence score: {top5_prob[i].item()}')
        # TODO: add batch inference

# hugging face model path default is .cache/huggingface/hub/xx.pth


model = InceptionV4()
model.preprocess()
model.inference()
