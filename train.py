import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import my_datasets
from model import mymodel

max_epoch = 30

if __name__ == '__main__':
    train_datas=my_datasets.mydatasets("./dataset/train")
    test_data=my_datasets.mydatasets("./dataset/test")
    train_dataloader=DataLoader(train_datas,batch_size=64,shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    m=mymodel().cuda()
    loss_fn=nn.MultiLabelSoftMarginLoss().cuda()
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001)
    w=SummaryWriter("logs")
    total_step=0

    for epoch in range(max_epoch):
        print("外层训练次数{}".format(epoch))
        for i,(imgs,targets) in enumerate(train_dataloader):
            imgs=imgs.cuda()
            targets=targets.cuda()
            outputs=m(imgs)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%100==0:
                total_step+=1
                print("训练{}次,loss:{}".format(total_step*10, loss.item()))
                w.add_scalar("loss",loss,total_step)

            # writer.add_images("imgs", imgs, i)
            # tensorboard --logdir=logs

    torch.save(m, "./checkpoints/model.pth")
