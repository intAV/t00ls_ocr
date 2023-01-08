
import random
import time

import requests
from PIL import Image
from torch.utils.data import DataLoader
from one_hot import captcha_array,vec2text
import torch
import my_datasets
from torchvision import transforms
from io import BytesIO

headers = {
        "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    }
session = requests.session()

model_path="./checkpoints/model.pth"

#批量验证测试集
def test_pred():
    m = torch.load(model_path).cuda()
    m.eval()
    test_data = my_datasets.mydatasets("./dataset/aaa")

    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    test_length = test_data.__len__()
    correct = 0;
    for i, (imgs, lables) in enumerate(test_dataloader):
        imgs = imgs.cuda()
        lables = lables.cuda()

        lables = lables.view(-1, captcha_array.__len__())

        lables_text = vec2text(lables)
        predict_outputs = m(imgs)
        predict_outputs = predict_outputs.view(-1, captcha_array.__len__())
        predict_labels = vec2text(predict_outputs)
        if predict_labels == lables_text:
            correct += 1
            print("预测正确：正确值:{},预测值:{}".format(lables_text, predict_labels))
        else:
            print("预测失败:正确值:{},预测值:{}".format(lables_text, predict_labels))
        # m(imgs)
    print("正确率{}".format(correct / test_length * 100))

#单张图片的预测
def pred_pic(pic_path):
    img = Image.open(pic_path)
    print(img)
    tersor_img = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((50, 100)),
        transforms.ToTensor()
    ])
    img = tersor_img(img).cuda()
    img = torch.reshape(img, (-1, 1, 50, 100))
    m = torch.load(model_path).cuda()
    outputs = m(img)
    outputs = outputs.view(-1, len(captcha_array))
    outputs_lable = vec2text(outputs)
    print(outputs_lable)

#通过WEB验证
def net_img_pic(session):
    update = "".join(random.sample([str(x) for x in range(10)], 4))
    url = "https://www.t00ls.com/seccode.php?update=" + update
    print(url)
    headers["Referer"] = "https://www.t00ls.com/"
    res = session.get(url, headers=headers)
    BytesIOObj = BytesIO()
    BytesIOObj.write(res.content)
    img = Image.open(BytesIOObj)
    #img.show()
    tersor_img = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((50, 100)),
        transforms.ToTensor()
    ])
    img = tersor_img(img).cuda()
    img = torch.reshape(img, (-1, 1, 50, 100))
    m = torch.load(model_path).cuda()
    outputs = m(img)
    outputs = outputs.view(-1, len(captcha_array))
    outputs_lable = vec2text(outputs)
    time.sleep(0.1)
    post_img_code(outputs_lable,res.content)


def post_img_code(code,img_data):
    post_url = "https://www.t00ls.com/domain.html"
    post_data = {
        "domain": "ji28.com",
        "formhash": "b7fee719",
        "querydomainsubmit": "查询",
        "seccodeverify": code,
    }
    print(post_data)
    try:
        res = session.post(url=post_url,
                           headers=headers,
                           data=post_data,
                           timeout=5)
        res_text = res.text
        if "Error:验证码不正确！" in res_text:
            print("Error:验证码不正确！")
        elif "ji28.com" in res_text:
            print("验证码正确,保存图片!...")
            with open("./dataset/train/" + code + '.jpg', 'wb') as f:
                f.write(img_data)
    except:
        print("!!! post_domain Error !!!")


if __name__ == '__main__':
    #test_pred();
    for i in range(100):
        print(i)
        net_img_pic(session)
    #pred_pic("./dataset/test/EEG6.jpg")

