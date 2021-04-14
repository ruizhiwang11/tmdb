import torch
from torch import optim, nn
import visdom
import torchvision
import PIL
from torch.utils.data import DataLoader
from torchvision import transforms

from Poster import Poster
from torchvision.models import resnet18

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

batchsz = 128
lr = 0.0003
epochs = 1000

device = torch.device('cuda')
torch.manual_seed(1234)

transform_train = transforms.Compose([
    lambda x: PIL.Image.open(x).convert('RGB'),  # string path= > image data
    transforms.Resize((int(224*1.25), int(224*1.25))),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    lambda x: PIL.Image.open(x).convert('RGB'),  # string path= > image data
    transforms.Resize((int(224), int(224))),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])
train_db = Poster(transform_train, mode='train')
val_db = Poster(transform_test, mode='validation')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,
                          num_workers=4)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=2)

viz = visdom.Visdom()


def evalute(model, loader, epoch):
    model.eval()

    correct = 0
    total = len(loader.dataset)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    accuracy = correct / total
    print(f"epoch {epoch}: validation accuracy is {accuracy *100}%")

    return correct / total


def main():
    # model = ResNet18(5).to(device)
    trained_model = resnet18(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1],  # [b, 512, 1, 1]
                          Flatten(),  # [b, 512, 1, 1] => [b, 512]
                          nn.Linear(512, 256),
                          nn.ReLU(),
                          nn.BatchNorm1d(256),
                          nn.Dropout(0.3),
                          nn.Linear(256,2),
                          # nn.ReLU(),
                          # nn.BatchNorm1d(32),
                          # nn.Dropout(0.1),
                          # nn.Linear(32,2),
                          # nn.LogSoftmax(dim=1)
                          ).to(device)


    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    for epoch in range(epochs):

        for step, (x, y) in enumerate(train_loader):
            # x: [b, 3, 224, 224], y: [b]
            x, y = x.to(device), y.to(device)

            model.train()
            logits = model(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(f"loss is {loss.item()}")
            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1

        if epoch % 1 == 0:

            val_acc = evalute(model, val_loader, epoch)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model.state_dict(), 'SGD-moment-best-1.mdl')

                viz.line([val_acc], [global_step], win='val_acc', update='append')

    print('best acc:', best_acc, 'best epoch:', best_epoch)

    # model.load_state_dict(torch.load('best.mdl'))
    # print('loaded from ckpt!')
    #
    # test_acc = evalute(model, test_loader)
    # print('test acc:', test_acc)


if __name__ == '__main__':
    main()