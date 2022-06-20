from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

PATH_TO_EVENTS = '/home/dejavu/datasets/Events_ImageSanitization/{}'
EVENTS = ["NationalMuseum", "BangladeshFire", "NotreDame", "Grenfell", "BostonMarathon"]
PATH_TO_FT = "../../events_features/Resnet50-ImageNet/{}.ft"

GPU = 4

model = torch.load("../../artifacts/models/protest_model_best.pth.tar")

device = torch.device("cuda:4")
model.to(device)

num_ftrs = model.fc.in_features

my_embedding = torch.zeros(num_ftrs)
my_embedding.to(device)

def copy_data(m, i, o):
    my_embedding.copy_(o.data.squeeze())

layer = model._modules.get('avgpool')
layer.register_forward_hook(copy_data)


transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

for e in tqdm(EVENTS, leave=False):
    dataset = ImageFolder(PATH_TO_EVENTS.format(e),transform=transform)
    
    dtl = DataLoader(dataset)
    
    model.eval()
    
    features = np.zeros([len(dataset.samples),num_ftrs])
    label = np.zeros([len(dataset.samples)], dtype=np.int8)
    src = np.zeros([len(dataset.samples)], dtype=object)
    
    for i, data in tqdm(enumerate(dtl)):
        model(data[0].to(device))
    
        features[i] = np.copy(my_embedding)
        src[i] = dtl.dataset.imgs[i][0]
        label[i] = dtl.dataset.imgs[i][1]
    
    df = pd.DataFrame({
        "feature_vector":[f for f in features],
        "src": src,
        "label": label
    })

    df.to_feather(PATH_TO_FT.format(e))