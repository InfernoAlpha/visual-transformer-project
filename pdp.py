import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms,datasets
from torch.utils.data import DataLoader

transform1 = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform2 = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


#train_dataset = datasets.ImageFolder(r"New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train",transform=transform1)
#test_dataset = datasets.ImageFolder(r"New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid",transform=transform1)

#train_dataloder = DataLoader(train_dataset,32,shuffle=True)
#test_dataloder = DataLoader(test_dataset,32,shuffle=False)


class N_conv(nn.Module):
    def __init__(self,in_channels,out_channels,N = 2):
        super(N_conv,self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels,out_channels,kernel_size=(3,3),padding=(1,1)))
        model.append(nn.ReLU(True))
        for i in range(N-1):
            model.append(nn.Conv2d(out_channels,out_channels,kernel_size=(3,3),padding=(1,1)))
            model.append(nn.ReLU(True))
        model.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
        self.conv = nn.Sequential(*model)
    
    def forward(self,x):
        return self.conv(x)
    
class Vgg16(nn.Module):
    def __init__(self,in_channels,out_channels,init_weights=True):
        super(Vgg16,self).__init__()
        self.conv1 = N_conv(in_channels,64)
        self.conv2 = N_conv(64,128)
        self.conv3 = N_conv(128,256,N=2)
        self.conv4 = N_conv(256,512,N=2)
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.linear1 = nn.Linear(512*7*7,1024)
#        self.linear2 = nn.Linear(4096,4096)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.4)
        self.liner3 = nn.Linear(1024,out_channels)
        self.softmax = nn.Softmax()
        if init_weights:
            self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,mode = 'fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)
    def forward(self,x):
        return self.softmax(self.liner3(self.dropout(self.relu(self.linear1(torch.flatten(self.avgpool(self.conv4(self.conv3(self.conv2(self.conv1(x))))),1))))))
         
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 7

    model = Vgg16(3,38).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = 1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5) 

    def check_accuracy(loader,model):
        num_correct = 0
        num_samples = 0
        model.eval()

        with torch.no_grad():
            for x,y in loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                _, pred = logits.max(1)
                num_correct += (pred == y).sum()
                num_samples += pred.size(0)
        model.train()
        return num_correct/num_samples
    
    for epoch in range(epochs):
        loop = tqdm(enumerate(train_dataloder),total=len(train_dataloder))
        for batch_idx,(data,targets) in loop:
            data = data.to(device)
            target = targets.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = loss_fn(pred,target)
            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch [{epoch+1}]')
            loop.set_postfix(loss = loss.item())
        scheduler.step(check_accuracy(test_dataloder,model))
    print(check_accuracy(test_dataloder, model))

    save_path = Path("model_MP_aws_2.pth")
    torch.save(model.state_dict(),save_path)