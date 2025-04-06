import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

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
        self.conv3 = N_conv(128,256,N=3)
        self.conv4 = N_conv(256,512,N=3)
        self.conv5 = N_conv(512,512,N=3)
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.linear1 = nn.Linear(512*7*7,4096)
        self.linear2 = nn.Linear(4096,4096)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.3)
        self.liner3 = nn.Linear(4096,out_channels)
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
        x = self.liner3(self.dropout(self.relu(self.linear2(self.dropout(self.relu(self.linear1(torch.flatten(self.avgpool(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))),1))))))))
        return x
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 20

    model = Vgg16(3,38).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = 1e-4)

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

    print(check_accuracy(test_dataloder, model))

    save_path = Path("model_MP6.pth")
    torch.save(model.state_dict(),save_path)