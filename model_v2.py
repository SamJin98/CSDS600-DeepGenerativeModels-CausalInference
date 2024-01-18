
import numpy as np
import torch
import torch.nn.functional as F
import utils as ut
from torch import autograd, nn, optim
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
import math  

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

#THis code is for CausalVAE and some improments to it.
#SEnet
class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        #y is the weight of every features.
        return x * y.expand_as(x)
    
    
#Encoder
#the Encoder here is used for dataset celeba.
class BN_Conv2d(nn.Module):
    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 groups=1, bias=False, activation=True) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.SiLU())
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides, is_se=True):
        super(BasicBlock, self).__init__()
        self.is_se = is_se
        self.conv1 = BN_Conv2d(in_channels, out_channels, 3, stride=strides, padding=1, bias=False)  # same padding
        self.conv2 = BN_Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, activation=False)
        if self.is_se:
            self.se = SELayer(out_channels)

        # fit input with residual output
        self.short_cut = nn.Sequential()
        if strides is not 1:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=strides, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
                )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.is_se:
            out =self.se(out)
        out = out + self.short_cut(x)
        return F.silu(out)
    
class Encoder(nn.Module):
    def __init__(self, z_dim, channel=4, y_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.channel = channel
        #here we change self.channel*96*96 to 218*178*3
        self.mean = nn.Linear(z_dim, z_dim)
        self.variance = nn.Linear(z_dim, z_dim)

        self.conv=nn.Sequential(BasicBlock(3,16,2),BasicBlock(16,32,2),BasicBlock(32,64,2),BasicBlock(64,128,2))

        self.linear=nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*14*12,1024),
            nn.ELU(),
            nn.Linear(1024,256),
            nn.ELU(),
            nn.Linear(256,z_dim)
        )

    def encode(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        #xy = xy.view(-1, 218*178*3)
        h0 = self.conv(xy)
        h=self.linear(h0)
        
        m, v = self.mean(h), self.variance(h)
        #print(self.z_dim,m.size(),v.size())
        return m, v
    
#The DAG layer
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

        x = torch.matmul(self.B.t(), x)
        return x
        
    def mask_u(self,x):
        #label u
        self.B = self.A
        
        x=torch.squeeze(x).float()
        #print(self.B)
        x = torch.matmul(self.B.t(), x.t()).t()
        return x


    def calculate_dag(self, x, v):        
        if x.dim()>2:
            x = x.permute(0,2,1)
        x = F.linear(x, torch.inverse(self.I - self.A.t()), self.bias) 
        #print(x.size())
       
        if x.dim()>2:
            x = x.permute(0,2,1).contiguous()
        return x,v
        

        
    def calculate_gaussian_ini(self, x, v):
       #print(self.A)

        if x.dim()>2:
            x = x.permute(0,2,1)
            v = v.permute(0,2,1)
        x = F.linear(x, torch.inverse(self.I - self.A), self.bias)
        v = F.linear(v, torch.mul(torch.inverse(self.I - self.A),torch.inverse(self.I - self.A)), self.bias)
        if x.dim()>2:
            x = x.permute(0,2,1).contiguous()
            v = v.permute(0,2,1).contiguous()
        return x, v
 

    #def encode_
    def forward(self, x):
        x = x * torch.inverse((self.A)+self.I)
        return x
    
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


#Decoder
#The decoder here is used for dataset celeba.
class Decoder_con(nn.Module):
    def __init__(self, in_channels, out_channels, strides, is_se=True,upsample=True):
        super().__init__()
        self.upsample=upsample
        self.net=nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.is_se = is_se
        self.conv1 = BN_Conv2d(in_channels, out_channels, 3, stride=strides, padding=1, bias=False)  # same padding
        self.conv2 = BN_Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, activation=False)
        if self.is_se:
            self.se = SELayer(out_channels)

        # fit input with residual output
        self.short_cut = nn.Sequential()
        if strides is not 1:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=strides, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
                )

    def forward(self, x):
        out = self.conv1(x)
        if self.upsample:
            out=self.net(out)
            skip=self.short_cut(out)
            
        out = self.conv2(out)
        if self.is_se:
            out =self.se(out)

        
        if self.upsample:
            out = out + skip
        else:
            out=out+self.short_cut(x)
        return F.silu(out)

class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim

        self.net1 = Decoder_con(1,3,1,is_se=False)
        self.net2= Decoder_con(3,3,1,is_se=False,upsample=False)
        self.linear=nn.Sequential(nn.Linear(z_dim,1024),
                                  nn.ELU(),
                                  nn.Linear(1024,4096),
                                  nn.ELU(),
                                  nn.Linear(4096,109*89)
                                  )
  
    def decode(self, z):
        z0=self.linear(z)
        z1=z0.view(z0.shape[0],1,109,89)
        z2=self.net1(z1)
        z3=self.net2(z2)        
  
        return z3
   