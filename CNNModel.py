import torch
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.conv_unit = nn.Sequential(
           # x: [b, 3, 32, 32] => [b, 16, ]
           nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0),
           nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
           #
           nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
           nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
           #
       )

       self.fc_unit = nn.Sequential(
           nn.Linear(32 * 5 * 5, 32),
           nn.ReLU(),
           # nn.Linear(120, 84),
           # nn.ReLU(),
           nn.Linear(32, 2)
       )

   def forward(self, x):
       """
       :param x: [b, 3, 32, 32]
       :return:
       """
       batchsz = x.size(0)
       # [b, 3, 32, 32] => [b, 16, 5, 5]
       x = self.conv_unit(x)
       # [b, 16, 5, 5] => [b, 16*5*5]
       x = x.view(batchsz, 32 * 5 * 5)
       # [b, 16*5*5] => [b, 10]
       logits = self.fc_unit(x)

       # # [b, 10]
       # pred = F.softmax(logits, dim=1)
       # loss = self.criteon(logits, y)

       return logits
