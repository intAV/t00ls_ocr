
import torch

captcha_array = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
captcha_size = 4

def text2vec(text):
    vectors=torch.zeros((captcha_size,captcha_array.__len__()))
    text = text.split(".")[0]
    text = text.upper()
    for i in range(len(text)):
        vectors[i,captcha_array.index(text[i])]=1
    return vectors

def vec2text(vec):
    vec=torch.argmax(vec,dim=1)
    text_label=""
    for v in vec:
        text_label+=captcha_array[v]
    return  text_label

if __name__ == '__main__':
    vec=text2vec("ACSV")
    #vec=vec.view(1,-1)[0]
    print(vec,vec.shape)
    print(vec2text(vec))