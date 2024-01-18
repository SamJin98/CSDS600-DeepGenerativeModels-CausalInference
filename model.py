import numpy as np
import torch
import torch.nn.functional as F
import utils as ut
from torch import autograd, nn, optim
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

class MaskLayer(nn.Module):
    def __init__(self, z_dim, concept=4,z2_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.z2_dim = z2_dim
        self.concept = concept
        
        self.elu = nn.ELU()
        self.net1 = nn.Sequential(
            nn.Linear(z2_dim , 128),
            nn.ELU(),
            nn.Linear(128, z2_dim)
        )
        self.net2 = nn.Sequential(
            nn.Linear(z2_dim , 128),
            nn.ELU(),
            nn.Linear(128, z2_dim)
        )
        self.net3 = nn.Sequential(
           nn.Linear(z2_dim , 128),
            nn.ELU(),
            nn.Linear(128, z2_dim)
        )
        self.net4 = nn.Sequential(
            nn.Linear(z2_dim , 128),
            nn.ELU(),
            nn.Linear(128, z2_dim)
        )
   
    def mix(self, z):
        zy = z.view(-1, self.concept*self.z2_dim)
        if self.z2_dim == 1:
            zy = zy.reshape(zy.size()[0],zy.size()[1],1)
            if self.concept ==4:
                zy1, zy2, zy3, zy4= zy[:,0],zy[:,1],zy[:,2],zy[:,3]
            elif self.concept ==3:
                zy1, zy2, zy3= zy[:,0],zy[:,1],zy[:,2]
        else:
            if self.concept ==4:
                zy1, zy2, zy3, zy4 = torch.split(zy, self.z2_dim, dim = 1)
            elif self.concept ==3:
                zy1, zy2, zy3= torch.split(zy, self.z2_dim, dim = 1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        if self.concept ==4:
            rx4 = self.net4(zy4)
            h = torch.cat((rx1,rx2,rx3,rx4), dim=1)
        elif self.concept ==3:
            h = torch.cat((rx1,rx2,rx3), dim=1)
        #print(h.size())
        return h



class Attention(nn.Module):
    def __init__(self, in_features, bias=False):
        super().__init__()
        self.M =  nn.Parameter(torch.nn.init.normal_(torch.zeros(in_features,in_features), mean=0, std=1))
        self.sigmd = torch.nn.Sigmoid()
        #self.M =  nn.Parameter(torch.zeros(in_features,in_features))
        #self.A = torch.zeros(in_features,in_features).to(device)
        
    def attention(self, z, e):
        a = torch.matmul(self.M,z)
        a=torch.matmul(a,e.permute(0,2,1))
        a = self.sigmd(a)
        #print(self.M)
        A = torch.softmax(a, dim = 1)
        e = torch.matmul(A,e)
        return e, A

    
class DagLayer(nn.Linear):
    def __init__(self, in_features, out_features,i = False, bias=False):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.i = i
        self.a = torch.zeros(out_features,out_features)
        self.a = self.a
        #self.a[0][1], self.a[0][2], self.a[0][3] = 1,1,1
        #self.a[1][2], self.a[1][3] = 1,1
        self.A = nn.Parameter(self.a)
        
        self.b = torch.eye(out_features)

        self.b = self.b
        self.B = nn.Parameter(self.b)
        
        self.I = nn.Parameter(torch.eye(out_features))
        self.I.requires_grad=False
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
    def mask_z(self,x):
        self.B = self.A
        #if self.i:
        #    x = x.view(-1, x.size()[1], 1)
        #    x = torch.matmul((self.B+0.5).t().int().float(), x)
        #    return x
        x = torch.matmul(self.B.t(), x)
        return x
        
    def mask_u(self,x):
        self.B = self.A
        #if self.i:
        #    x = x.view(-1, x.size()[1], 1)
        #    x = torch.matmul((self.B+0.5).t().int().float(), x)
        #    return x
        
        #x = x.view(-1, x.size()[1], 1)
        x=torch.squeeze(x).float()
        #print(self.B)
        x = torch.matmul(self.B.t(), x.t()).t()
        return x
        
    def inv_cal(self, x,v):
        if x.dim()>2:
            x = x.permute(0,2,1)
        x = F.linear(x, self.I - self.A, self.bias)
       
        if x.dim()>2:
            x = x.permute(0,2,1).contiguous()
        return x,v

    def calculate_dag(self, x, v):
        #print(self.A)
        #x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)
        
        if x.dim()>2:
            x = x.permute(0,2,1)
        x = F.linear(x, torch.inverse(self.I - self.A.t()), self.bias) 
        #print(x.size())
       
        if x.dim()>2:
            x = x.permute(0,2,1).contiguous()
        return x,v
    
    
#encode
#nvae cell structure
class ResidualCell(nn.Module):
  def __init__(self, channels):
    super(ResidualCell, self).__init__()
    self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(channels)
    self.relu = nn.ReLU()
    self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(channels)

  def forward(self, x):
    identity = x

    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))

    out += identity
    out = self.relu(out)

    return out
  
class Encoder_net(nn.Module):
  def __init__(self, channels=[32, 64, 128], latent_dim=128, kernel_size=3):
    super(Encoder_net, self).__init__()
    self.initial_conv = nn.Conv2d(3, channels[0], kernel_size=kernel_size, stride=2, padding=1)
    self.residual_cells = nn.Sequential(
        ResidualCell(channels[0]),
        nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size, stride=2, padding=1),
        ResidualCell(channels[1])
    )
    self.final_conv = nn.Conv2d(channels[1], channels[2], kernel_size=kernel_size, stride=2, padding=1)
    self.flatten = nn.Flatten()
    self.fc = nn.Sequential(nn.Linear(channels[2]*644, 1024),
                            nn.LeakyReLU(0.2),
                            nn.Linear(1024, latent_dim*2))

  def forward(self, x):
    x = F.relu(self.initial_conv(x))
    x = self.residual_cells(x)
    x = F.relu(self.final_conv(x))
    
    x = self.flatten(x)
    x = self.fc(x)
    return x


class Encoder(nn.Module):
    def __init__(self, z_dim, channel=4, y_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.channel = channel
        #here we change self.channel*96*96 to 218*178*3
        self.fc1 = nn.Linear(218*178*3, 300)
        self.fc2 = nn.Linear(300+y_dim, 300)
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, 2 * z_dim)
        self.LReLU = nn.LeakyReLU(0.2, inplace=True)
        self.mean = nn.Linear(z_dim, z_dim)
        self.variance = nn.Linear(z_dim, z_dim)
        '''self.net=nn.Sequential(
            nn.Conv2d(3,8,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Conv2d(8,16,3,2,1),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16,32,3,2,1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(20608,1024),
            nn.ELU(),
            nn.Linear(1024,128),
            nn.ELU(),
            nn.Linear(128,2*z_dim)
        )'''
        self.net=Encoder_net(channels=[32, 64, 128], latent_dim=z_dim)

    def conditional_encode(self, x, l):
        x = x.view(-1, 218*178*3)
        x = F.elu(self.fc1(x))
        l = l.view(-1, 4)
        x = F.elu(self.fc2(torch.cat([x, l], dim=1)))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        m, v = ut.gaussian_parameters(x, dim=1)
        return m,v

    def encode(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)

        h = self.net(xy)
        m, v = ut.gaussian_parameters(h, dim=1)
        #m, v = self.mean(h), self.variance(h)

        return m, v
   
   
class Decoder(nn.Module):
    def __init__(self, z_dim, concept, z1_dim, channel = 4, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.concept = concept
        self.y_dim = y_dim
        self.channel = channel
        #print(self.channel)
        self.elu = nn.ELU()
        self.net1 = nn.Sequential(
            nn.Linear(z1_dim + y_dim, 1024),
            nn.ELU(),
            nn.Linear(1024, 218*178*3),
            nn.ELU()
        )
        self.convnet1=nn.Sequential(
            nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(3),
            nn.ELU(),
            nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(3),
            nn.ELU()
            )
        self.net2 =  nn.Sequential(
            nn.Linear(z1_dim + y_dim, 1024),
            nn.ELU(),
            nn.Linear(1024, 218*178*3),
            nn.ELU()
        )
        self.convnet2=nn.Sequential(
            nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(3),
            nn.ELU(),
            nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(3),
            nn.ELU()
            )
        self.net3 =  nn.Sequential(
            nn.Linear(z1_dim + y_dim, 1024),
            nn.ELU(),
            nn.Linear(1024, 218*178*3),
            nn.ELU()
        )
        self.convnet3=nn.Sequential(
            nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(3),
            nn.ELU(),
            nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(3),
            nn.ELU(),
            )
        self.net4 =  nn.Sequential(
            nn.Linear(z1_dim + y_dim, 1024),
            nn.ELU(),
            nn.Linear(1024, 218*178*3),
            nn.ELU()
        )
        '''nn.Sequential(
            nn.Linear(z1_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.ELU(),
            nn.Linear(1024, 218*178*3)
        )'''
        self.convnet4=nn.Sequential(
            nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(3),
            nn.ELU(),
            nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(3),
            nn.ELU(),
            )
        self.flatten=nn.Flatten()
        self.net5 = nn.Sequential(
            nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(3),
            nn.ELU()
        )
   

   
    def decode_union(self, z, y=None):
        
        z = z.view(-1, self.concept*self.z1_dim)
        zy = z if y is None else torch.cat((z, y), dim=1)
        if self.z1_dim == 1:
            zy = zy.reshape(zy.size()[0],zy.size()[1],1)
            zy1, zy2, zy3, zy4 = zy[:,0],zy[:,1],zy[:,2],zy[:,3]
        else:
            zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim//self.concept, dim = 1)
        rx1 = self.convnet1(self.net1(zy1).view(-1,3,218,178))
        rx2 = self.convnet2(self.net2(zy2).view(-1,3,218,178))
        rx3 = self.convnet3(self.net3(zy3).view(-1,3,218,178))
        rx4 = self.convnet4(self.net4(zy4).view(-1,3,218,178))

        h = (rx1+rx2+rx3+rx4)/4
        h=self.flatten(h)
        return h
   

    
    #modify self.net1, self。net2,self.net3, self.net4 to improve decoder
    def decode_sep(self, z,y=None):
        #z.shape =(batchsize, 16)
        z = z.view(-1, self.concept*self.z1_dim)
        zy = z if y is None else torch.cat((z, y), dim=1)
            
        if self.z1_dim == 1:
            zy = zy.reshape(zy.size()[0],zy.size()[1],1)
            if self.concept ==4:
                zy1, zy2, zy3, zy4= zy[:,0],zy[:,1],zy[:,2],zy[:,3]
            elif self.concept ==3:
                zy1, zy2, zy3= zy[:,0],zy[:,1],zy[:,2]
        else:
            if self.concept ==4:
                zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim//self.concept, dim = 1)
            elif self.concept ==3:
                zy1, zy2, zy3= torch.split(zy, self.z_dim//self.concept, dim = 1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        if self.concept ==4:
            rx4 = self.net4(zy4)
            #return average of 4 small decoders(zi) result where zi is the ith latent vector 
            h = (rx1+rx2+rx3+rx4)/self.concept
        elif self.concept ==3:
            h = (rx1+rx2+rx3)/self.concept
        
        return h
   
    def decode_cat(self, z, u, y=None):
        z = z.view(-1, 4*4)
        zy = z if y is None else torch.cat((z, y), dim=1)
        zy1, zy2, zy3, zy4 = torch.split(zy, 1, dim = 1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        rx4 = self.net4(zy4)
        h = self.net5( torch.cat((rx1,rx2, rx3, rx4), dim=1))
        return h