from tmdbv3api import Movie
from tmdbv3api import TMDb
from PIL import Image
import urllib3
import requests
import io
import os
import time
import pandas as pd
import numpy as np


tmdb = TMDb()
tmdb.api_key = '7f3920d45d516940ceaae0035522baf4'
image_url_prefix = "https://image.tmdb.org/t/p/w500/"
movie = Movie()

def grab_poster_tmdb(folder ,poster_path):

    url=image_url_prefix+str(poster_path) + ".jpg"
    strcmd ='wget -O '+ folder + str(poster_path) + ".jpg" + ' '+url
    os.system(strcmd)

# is_thriller_list = []
# image_id_list = []
#
# movie = Movie()
# output = pd.DataFrame()
# for i in range(1,501):
#     popular = movie.popular(page=i)
#     for p in popular:
#         if output.empty:
#             output = pd.DataFrame.from_dict(dict(p))
#         else:
#             output = output.append(dict(p), ignore_index=True)
#     print(output.head())
# output.to_csv("123.csv")

df = pd.read_csv("123.csv")

def label(x):
    if "27" in x:
        if x[1:3] == "27":
            return 0
        else:
            return 1
    else:
        return 2

# header = ["genre_ids", "poster_path", "title","id"]
# df = df[header]
# df['isThriller'] = df['genre_ids'].apply(label)
# print(df.head(100))
# df["poster_path"] = df["poster_path"].str.slice(start=1,stop=-4)
# df = df.drop([1,2,3,4]).reset_index()
# del df["index"]
# del df["genre_ids"]
# df = df.dropna().reset_index()
# del df["index"]
#
# df.to_csv("label_multi.csv")

df = pd.read_csv("label_multi.csv")
train_df = df.iloc[:8500,:]
test_df = df.iloc[8501:,:]
print(train_df["isThriller"].value_counts())
print(test_df["isThriller"].value_counts())
train_df[["poster_path","isThriller","id"]].to_csv("train_multi.csv",index=False,header=None)
test_df[["poster_path","isThriller","id"]].to_csv("test_multi.csv",index=False,header=None)
train_poster_list = train_df["poster_path"].tolist()
test_poster_list = test_df["poster_path"].tolist()

# for poster in train_poster_list:
#     grab_poster_tmdb("./train/", poster)
# for poster in test_poster_list:
#     grab_poster_tmdb("./test/", poster)

# import pandas as pd
# import PIL
# import torch
# from torch.autograd import Variable
# import torch.nn.functional as F
# from torchvision import transforms
# from torch.utils.data.dataset import Dataset
# from torch.utils.data import DataLoader
# from torch import nn
# from torchvision.models import resnet18
# from Traintransferlearning import Flatten
#
# imsize = 224
# loader = transforms.Compose([transforms.Resize(imsize), transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                      std=[0.229, 0.224, 0.225])])
#
# def image_loader(image_name):
#     """load image, returns cuda tensor"""
#     image = Image.open(image_name).convert('RGB')
#     image = loader(image).float()
#     image = Variable(image, requires_grad=True)
#     image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
#     return image.cuda()  #assumes that you're using GPU
#
# movie_id = input("Please enter a movie id:")
# movie = Movie()
# m = dict(movie.details(movie_id))
#
# poster_path = m["poster_path"][1:-4]
# genres = m["genres"]
#
# print(poster_path)
# print(genres)
# grab_poster_tmdb("./evaluation/", poster_path)
#
# image = image_loader("./evaluation/"+poster_path+".jpg")
#
# trained_model = resnet18(pretrained=True)
# model = nn.Sequential(
#     *list(trained_model.children())[:-1],  # [b, 512, 1, 1]
#     Flatten(),  # [b, 512, 1, 1] => [b, 512]
#     nn.Linear(512, 128),
#     nn.ReLU(),
#     nn.BatchNorm1d(128),
#     nn.Dropout(0.2),
#     nn.Linear(128, 32),
#     nn.ReLU(),
#     nn.BatchNorm1d(32),
#     nn.Dropout(0.1),
#     nn.Linear(32, 2),
#     nn.LogSoftmax(dim=1)
#     ).to("cuda")
# model.load_state_dict(torch.load("SGD-moment-best.mdl"))
# model.eval()
# prediction = torch.tensor([])
# with torch.no_grad():
#     prediction = model(image)
#
# print(prediction.argmax(dim=1).cpu().detach().numpy())
