

from logging import root
import torch
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
import glob



class Model(torch.nn.Module):

    def __init__(self,in_dim,h_dim):

        super(Model, self).__init__()
        self.layer1 = torch.nn.Linear(in_dim,h_dim)
        self.layer2 = torch.nn.Linear(h_dim,4)

    def forward(self,x):

        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)

        return x

def collate(inp):
    #print(inp[0].shape,inp[1].shape)
    return torch.cat(inp,dim=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


dataset  = [torch.load(i,map_location=device) for i in glob.glob("/projects/iiitd/mrcnn/pretrained/rpndata/*")]
data = collate(dataset)
data = data.to(device)


model = Model(8,16).to(device)
#model= torch.nn.DataParallel(model)
print(model)
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

x,y = data[:,4:12], data[:,12:]

for e in range(1000):

    out = model(x)
    out = torch.tanh(out)*x[:,4:]
    optimizer.zero_grad()

    loss = criterion(out, y)
  
    loss.backward()
    optimizer.step()
    print("Epoch :",e," Loss : ",loss.item())



torch.save(model,'/projects/iiitd/mrcnn/pretrained/rpn.pth')
