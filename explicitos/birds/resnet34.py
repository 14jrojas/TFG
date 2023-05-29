import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import time
import logging
from datetime import datetime

import torchsummary

# para tener la misma referencia de medidas
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

# logger
log_filename = datetime.now().strftime("output/resnet34-birds_train_%Y_%m_%d_%H_%M.log")
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s,%(msecs)03d %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# cudnn
cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

# seed
torch.manual_seed(42)

# augmentation
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
transform_test = transforms.Compose([
            transforms.Resize(int(224 / 0.875)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

# loaders
data_dir = 'data'
train_dataset = datasets.ImageFolder(data_dir + '/train', transform = transform_train)
test_dataset = datasets.ImageFolder(data_dir + '/val', transform = transform_test)
train_loader = DataLoader(dataset=train_dataset,
                           batch_size=32*2, 
                           shuffle=True, 
                           num_workers=6,
                           pin_memory=True, 
                           drop_last=True)
test_loader = DataLoader(dataset=test_dataset,
                           batch_size=32*2,
                           shuffle=False,
                           num_workers=6,
                           pin_memory=True,
                           drop_last=True)

# model
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', weights=None, num_classes=525)

# loss, optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.90, weight_decay=0.00005)

# entrenando en gpus especificas
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.DataParallel(model)
model = model.to(device)

# lr scheduler
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, len(train_loader)*100, eta_min=1e-6)

# imprimir parametros del modelo
print(torchsummary.summary(model, (3,224,224)))

# check accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            loss = criterion(scores, targets)
            losses.update(loss.item(), data.size(0))

            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)

        accuracy = float(num_correct) / float(num_samples) * 100

    model.train()

    return losses.avg, accuracy


# training loop
num_epochs = 100
for epoch in range(num_epochs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    start_time = time.time()
    for batch_idx, (data, targets) in enumerate(train_loader):        
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        losses.update(loss.item(), data.size(0))

        # Compute batch statistics
        _, preds = scores.max(1)
        correct = (preds == targets).sum().item()
        samples = data.size(0)
        
        top1.update(correct*100/samples)
        
        batch_time.update(time.time() - start_time)
        start_time = time.time()
        
        torch.cuda.empty_cache()

        if batch_idx % 100 == 0:
            logging.info(f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}] \tTime {batch_time.avg:.3f}s\tSpeed {samples / batch_time.avg:.1f} samples/s\tLoss {losses.avg:.5f}\tAcc@1 {top1.avg:.3f}')
            print(f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}] \tTime {batch_time.avg:.3f}s\tSpeed {samples / batch_time.avg:.1f} samples/s\tLoss {losses.avg:.5f}\tAcc@1 {top1.avg:.3f}')

    epoch_loss, epoch_acc = check_accuracy(test_loader, model)
    torch.cuda.empty_cache()
    logging.info(f'Test: \tLoss {epoch_loss:.5f}\tAcc@1: {epoch_acc:.2f}')
    print(f'Test: \tLoss {epoch_loss:.5f}\tAcc@1: {epoch_acc:.2f}')
