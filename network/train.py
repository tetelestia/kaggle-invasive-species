"""
Docstring
"""
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from dataset import InvasiveDataset

from models.resnet import resnet18 as Net

img_size = (256, 256)
batch_size = 28
num_epochs = 10

train_dataset = InvasiveDataset(label_csv='../data/train_labels.csv',
                                img_path='../data/train/',
                                # transform=transforms.Compose([
                                    # transforms.Lambda(lambda x: randomBasicTransform(x)),
                                    # transforms.Lambda(lambda x: toTensor(x)),
                                # ]),
                                img_size=img_size)

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=3,
                          pin_memory=True)

model = Net(pretrained=True, num_classes=1).cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

criterion = nn.BCELoss().cuda()
# criterion = nn.CrossEntropyLoss().cuda()

try:
    for epoch in range(1, num_epochs+1):
        start = time.time()

        model.train()
        cumulative_loss = 0

        for batch_index, (images, target) in enumerate(train_loader):
            # Make image and output Torch Variables
            images, target = Variable(images).cuda(), Variable(target).cuda()
    
            # Zero model gradients
            optimizer.zero_grad()
    
            # Run data through model
            output = model(images)

            # Apply sigmoid to output to stabalize BCE Loss
            output = F.sigmoid(output)
    
            # Calculate loss
            loss = criterion(output, target)
            cumulative_loss += loss.data[0]
    
            # Backpropogate loss through model
            loss.backward()
    
            # Update model parameters
            optimizer.step()
    
            if batch_index % 10 == 0:
                print('Epoch: {} [{}/{} ({:.0f}%)]\t train_loss: {:.4f}'.format(
                      epoch, batch_index * len(images), len(train_loader.dataset),
                      100. * batch_index/len(train_loader), cumulative_loss/(batch_index+1)), end='\r')

except KeyboardInterrupt:
    pass    
