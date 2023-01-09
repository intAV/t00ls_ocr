import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import my_datasets
from resnet50.resnet import myresnet

max_epoch = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    train_datas=my_datasets.mydatasets("../dataset/train")
    test_data=my_datasets.mydatasets("../dataset/test")
    train_dataloader=DataLoader(train_datas,batch_size=126,shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    m=myresnet().to(device)
    loss_fn=nn.MultiLabelSoftMarginLoss().to(device)
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001)
    writer=SummaryWriter("logs")
    w=SummaryWriter("logs")
    total_step=0

for epoch in range(max_epoch):
    print("外层训练次数{}".format(epoch+1))
    for i,(imgs,targets) in enumerate(train_dataloader):
        imgs=imgs.to(device)
        targets=targets.to(device)
        outputs=m(imgs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_step+=1
        print("训练{}次,loss:{},次数:{}".format(total_step*100, loss.item(),i))
        w.add_scalar("loss",loss,total_step)

    writer.close()
    # tensorboard --logdir=logs

torch.save(m.state_dict(), "../checkpoints/res-model.pth")
