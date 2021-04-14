import torch
import os
import random, csv
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import PIL

class Poster(Dataset):

    def __init__(self,tranform, mode):
        super(Poster,self).__init__()
        self.tranform = tranform
        self.train_df = pd.read_csv('./train.csv')
        self.test_df = pd.read_csv('./test.csv')
        self.mode = mode
        if mode == "train":
            self.images, self.labels = self.load_train_csv("train.csv")
        if mode == "validation":
            self.images, self.labels = self.load_train_csv("test.csv")
    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        # idx~[0-len(images)]
        img, label = self.images[idx], self.labels[idx]

        tf = self.tranform
        img = tf(img)
        label = torch.tensor(label)


        return img, label

    def denormalize(self, x_hat):
        """To view to image without normalization"""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean

        return x

    def load_train_csv(self, filename):
        images_path,  labels = [], []
        with open(os.path.join("./", filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img_file_name, label, _ = row
                if self.mode == "train":
                    images_path.append(os.path.join( "./train", img_file_name+".jpg"))
                else:
                    images_path.append(os.path.join("./test", img_file_name + ".jpg"))
                # print(label)
                if label == "True":
                    labels.append(1)
                else:
                    labels.append(0)
        assert len(images_path) == len(labels)
        return images_path, labels




def main():
    import visdom
    import time
    import torchvision

    transform = transforms.Compose([
        lambda x: PIL.Image.open(x).convert('RGB'),  # string path= > image data
        transforms.Resize((int(224), int(224))),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    viz = visdom.Visdom()
    poster = Poster(transform, "validation")
    x, y = next(iter(poster))
    print('sample:', x.shape, y.shape, y)

    viz.image(poster.denormalize(x), win='sample_x', opts=dict(title='sample_x'))
    viz.image(x, win='sample_x', opts=dict(title='sample_x'))

    loader = DataLoader(poster, batch_size=32, shuffle=True, num_workers=8)

    for x, y in loader:
        viz.images(poster.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        #viz.image(poster.denormalize(x), win='sample_x', opts=dict(title='sample_x'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))

        # time.sleep(10)


if __name__ == '__main__':
    pass

    # main()