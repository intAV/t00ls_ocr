import torch
from torch import nn
from one_hot import captcha_array,captcha_size


class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer5 = nn.Sequential(
          nn.Flatten(),#[1, 9216]
          nn.Linear(in_features=512*3*6,out_features=4096),
          nn.Dropout(0.2),  # drop 20% of the neuron
          nn.ReLU(),
          nn.Linear(in_features=4096,out_features=captcha_size*captcha_array.__len__())
        )
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        #x = x.view(x.size(0), -1)
        return x;

if __name__ == '__main__':
    data=torch.ones(1,1,50,100)
    model=mymodel()
    x=model(data)
    print(x.shape)