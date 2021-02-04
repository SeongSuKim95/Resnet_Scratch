import torch
import torch.nn as nn
##Creating Basic Block

class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,identity_downsample = None, stride = 1):
         super(BasicBlock,self).__init__()

         self.expansion = 4
        
         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
         self.bn1 = nn.BatchNorm2d(out_channels)
         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1) # Maintain size
         self.bn2 = nn.BatchNorm2d(out_channels)
         self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size = 1, stride = 1, padding = 0)
         self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
         self.relu = nn.ReLU()
   
         self.identity_downsample = identity_downsample
          
         # Ex) [1x1 64 , 3x3 64, 1x1 256] x 3
         # conv1 = 1x1, conv2 = 3x3, conv3 = 1x1
         # identity_downsample = skip_connection

    def forward(self,x):

        # Residual block
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_downsample is not None:

            identity = self.identity_downsample(identity)
        
        x += identity
        x = self.relu(x)
        
        return x


class ResNet(nn.Module): #[3, 4, 6, 3]
    def __init__(self,block,layers,image_channels,num_classes):
        super(ResNet,self).__init__()
        self.in_channels = 64 # Initial Resnet block Input in_channels
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size = 7, stride = 2, padding =3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3,stride = 2,padding = 1)

        # ResNet layers
        self.layer1 = self._make_layers(block,layers[0],out_channels=64,stride = 1) # 64 64 256 Stride = 1
        self.layer2 = self._make_layers(block,layers[1],out_channels=128,stride = 2) # 128 128 512 Stride = 2
        self.layer3 = self._make_layers(block,layers[2],out_channels=256,stride = 2) # 256 256 1024 Stride = 2 
        self.layer4 = self._make_layers(block,layers[3],out_channels=512,stride = 2) # 512 512 2048 Stride = 2

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4,num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)

        x = self.fc(x)

        return x

    def _make_layers(self,block,num_residual_blocks,out_channels,stride): # block = Basicblock , num_residual_blocks = [3,4,6,3]

        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels,out_channels*4,kernel_size=1,stride = stride),
                                                nn.BatchNorm2d(out_channels*4))
        
        # Changing number of channels
        layers.append(BasicBlock(self.in_channels, out_channels, identity_downsample, stride)) #out_channels = 64
        self.in_channels = out_channels*4 # 256

        for i in range(num_residual_blocks-1):
            layers.append(block(self.in_channels,out_channels)) # 256 -> 64 , 64 *4 (256) again

        return nn.Sequential(*layers)


def ResNet50(img_channels = 3, num_classes = 1000):
    return ResNet(BasicBlock,[3,4,6,3],img_channels,num_classes)

def ResNet101(img_channels = 3, num_classes = 1000):
    return ResNet(BasicBlock,[3,4,23,3],img_channels,num_classes)

def ResNet152(img_channels = 3, num_classes = 1000):
    return ResNet(BasicBlock,[3,8,36,3],img_channels,num_classes)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparams

num_epochs = 20
batch_size = 100
learning_rate = 0.01


# Dataset
import torchvision
import torchvision.transforms as transforms

# Dataset Transform

transform_train = transforms.Compose([
                transforms.RandomCrop(32,padding = 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                                     ])


transform_test = transforms.Compose([
                transforms.ToTensor(),
                                    ]) 



train_dataset = torchvision.datasets.CIFAR10(root = "/home/sungsu21/Project/data",train = True, download = False, transform = transform_train)
test_dataset = torchvision.datasets.CIFAR10(root = "/home/sungsu21/Project/data",train = False, download = False, transform = transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = 128, shuffle = True, num_workers = 4)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = 100, shuffle = True, num_workers = 4)

# net
net = ResNet50(3,1000).to(device)

# Criterion
criterion = nn.CrossEntropyLoss()

# Optimizer

import torch.optim as optim

optimizer = optim.SGD(net.parameters(),lr = learning_rate, momentum = 0.9, weight_decay =  0.0002)

# Train the model

total_step = len(train_loader)

net.train()
for epoch in range(num_epochs):
        for i, (images,labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss  = criterion(outputs,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:

                print("Epoch [{}/{}], Step [{}/{}] Loss : {:.4f}".format(epoch+1,num_epochs,i+1,total_step,loss.item()))


net.eval()
with torch.no_grad():

    correct = 0 
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _ , predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    
    print('Accuaracy of the model on the test images: {}%'.format(100*correct/total))


