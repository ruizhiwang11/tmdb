import torch
from torch import optim, nn
import visdom
import torchvision
import PIL
from torch.utils.data import DataLoader
from torchvision import transforms
from CNNModel import Net
from Poster import Poster
from torchvision.models import resnet18


batchsz = 64
lr = 0.001
epochs = 100

device = torch.device('cuda')
torch.manual_seed(1234)

transform_train = transforms.Compose([
    lambda x: PIL.Image.open(x).convert('RGB'),  # string path= > image data
    transforms.Resize((int(32), int(32))),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    lambda x: PIL.Image.open(x).convert('RGB'),  # string path= > image data
    transforms.Resize((int(32), int(32))),
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
    model = Net()
    model.to(device)

    def weight_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)

    model.apply(weight_init)
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

                torch.save(model.state_dict(), 'mse-best-without-normal.mdl')

                viz.line([val_acc], [global_step], win='val_acc', update='append')

    print('best acc:', best_acc, 'best epoch:', best_epoch)

    # model.load_state_dict(torch.load('best.mdl'))
    # print('loaded from ckpt!')
    #
    # test_acc = evalute(model, test_loader)
    # print('test acc:', test_acc)


if __name__ == '__main__':
    main()