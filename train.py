import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils import data
import torch.utils.data as Data
from torchvision import transforms

import numpy as np
import math
import time
from utils2 import _h_A
import model

import matplotlib.pyplot as plt
import random
from PIL import Image
import os
import numpy as np

import utils as ut
import cv2
import re
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

#-----------------------------LOSS-----------------------------
def negative_elbo_bound(args,blocks,x,label):
    '''
    Computes the Evidence Lower Bound, KL and, Reconstruction costs
    Args:
        x: tensor: (batch, dim): Observations
    Returns:
        nelbo: tensor: (): Negative evidence lower bound
        kl: tensor: (): ELBO KL divergence to prior
        rec: tensor: (): ELBO Reconstruction term
    '''
    #encoder
    #q_m.shape=(batchsize,16)-->(batchsize,4,4)
    q_m, q_v = blocks['enc'].encode(x.to(device))
    q_m, q_v = q_m.reshape([q_m.size()[0], args.z1_dim,args.z2_dim]), torch.ones(q_m.size()[0], args.z1_dim,args.z2_dim).to(device)
    #dag
    #the result from encoder, according to the paper, we consider they follow a standard gaussian distribution
    decode_m, decode_v = blocks['dag'].calculate_dag(q_m.to(device), torch.ones(q_m.size()[0], args.z1_dim,args.z2_dim).to(device))
    decode_m, decode_v = decode_m.reshape([q_m.size()[0], args.z1_dim,args.z2_dim]),decode_v
    
    m_zm = blocks['dag'].mask_z(decode_m.to(device)).reshape([q_m.size()[0], args.z1_dim,args.z2_dim])
    m_zv=decode_v.reshape([q_m.size()[0], args.z1_dim,args.z2_dim])
    m_u = blocks['dag'].mask_u(label.to(device))
    
    f_z = blocks['mask_z'].mix(m_zm).reshape([q_m.size()[0], args.z1_dim,args.z2_dim]).to(device)
    e_tilde = blocks['attn'].attention(decode_m.to(device),q_m.to(device))[0]  
    f_z1 = f_z+e_tilde
    
    #mix label u
    g_u = blocks['mask_u'].mix(m_u).to(device)
    
    z_given_dag = ut.conditional_sample_gaussian(f_z1, m_zv*args.lambdav)
    #decoder
    decoded_bernoulli_logits = blocks['dec'].decode_union(z_given_dag.reshape([z_given_dag.size()[0], args.z_dim]))
    
    rec = ut.log_bernoulli_with_logits(x, decoded_bernoulli_logits.reshape(x.size()))
    rec = -torch.mean(rec)
    #calculate mean and varience
    p_m, p_v = torch.zeros(q_m.size()), torch.ones(q_m.size())
    cp_m, cp_v = ut.condition_prior(args.scale, label, args.z2_dim)
    cp_v = torch.ones([q_m.size()[0],args.z1_dim,args.z2_dim]).to(device)

    cp_z = ut.conditional_sample_gaussian(cp_m.to(device), cp_v.to(device))
    kl = torch.zeros(1).to(device)
    kl = args.alpha*ut.kl_normal(q_m.view(-1,args.z_dim).to(device), q_v.view(-1,args.z_dim).to(device), p_m.view(-1,args.z_dim).to(device), p_v.view(-1,args.z_dim).to(device))
    
    for i in range(args.z1_dim):
        kl = kl + args.beta*ut.kl_normal(decode_m[:,i,:].to(device), cp_v[:,i,:].to(device),cp_m[:,i,:].to(device), cp_v[:,i,:].to(device))
    kl = torch.mean(kl)
    mask_kl = torch.zeros(1).to(device) 

    for i in range(args.z1_dim):
        mask_kl = mask_kl + 1*ut.kl_normal(f_z1[:,i,:].to(device), cp_v[:,i,:].to(device),cp_m[:,i,:].to(device), cp_v[:,i,:].to(device))
    
    #all loss,here the reconstruction loss is MSE
    u_loss = torch.nn.MSELoss()
    label=torch.squeeze(label)
    mask_l = torch.mean(mask_kl) + u_loss(g_u, label.float().to(device))
    nelbo = 5*rec + kl + mask_l

    return nelbo, kl, rec, decoded_bernoulli_logits.reshape(x.size()), z_given_dag

def loss(x):
    nelbo, kl, rec = negative_elbo_bound(x)
    loss = nelbo

    summaries = dict((
        ('train/loss', nelbo),
        ('gen/elbo', -nelbo),
        ('gen/kl_z', kl),
        ('gen/rec', rec),
    ))

    return loss, summaries

def compute_sigmoid_given(blocks,z):
        logits = blocks['dec'].decode(z)
        return torch.sigmoid(logits)

def sample_z(args,batch,z_prior):
    return ut.sample_gaussian(z_prior[0].expand(batch,args.z_dim), z_prior[1].expand(batch, args.z_dim))

def sample_x( batch):
    z = sample_z(batch)
    return sample_x_given(z)

def sample_x_given(z):
    return torch.bernoulli(compute_sigmoid_given(z))

def sample_sigmoid(batch):
    z = sample_z(batch)
    return compute_sigmoid_given(z)

#---------------
def get_labels(list_attr_path,attr_list):
    labels=[]

    #get all labels (40)
    with open(list_attr_path) as f:
        lines=f.readlines()
        attrs=lines[1]
        result=re.split(' ',attrs)
        result.remove('\n')
        i=0

        for line in lines:
            if i>1:
                temp=re.split("\s",line)[1:]
                for t in temp:
                    if t=='':
                        temp.remove(t)
                temp=[float(x) for x in temp]
                labels.append(temp)
            i=i+1
    labels=np.array(labels)

    #get index of attr_list
    i=0
    index={}
    for a in result:
        if a in attr_list:
            index[a]=i
        i=i+1
    #print(index)

    #got 4 labels
    i=0
    for name in index:
        if i==0:
            temp=labels[:,index[name]]
            i=i+1
            continue
        temp=np.vstack([temp,labels[:,index[name]]])
        i=i+1
    #0-1
    temp[temp<0]=0
    labels_4=temp
    return labels_4

#---------------------dataset----------------------------
class dataload_withlabel(data.Dataset):
    def __init__(self, path, labels):
        self.path=path
        self.path_list=os.listdir(path)
        self.labels=labels
        

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        img=cv2.imread(self.path+'/'+self.path_list[index])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(img).permute(2, 0, 1)/255
        #img = self.transform(img)
        labels=self.labels[:,index]
        
        return image,labels.reshape(1,self.labels.shape[0])
    
#----------------------------------others----------------------------------
def _sigmoid(x):
    I = torch.eye(x.size()[0]).to(device)
    x = torch.inverse(I + torch.exp(-x))
    return x
    
class DeterministicWarmup(object):
    """
    Linear deterministic warm-up as described in
    """
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1/n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t
        return self.t

def save_model_by_name(model, global_step):
    save_dir = os.path.join('checkpoints', model.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))
 
#------------------------------------

#------------------------star train--------------

#parameters
class params():
    def __init__(self):
        self.z_dim=200
        self.z1_dim=4
        self.z2_dim=50
        self.inference = False 
        self.alpha=0.3
        self.beta=1
        self.lambdav=1e-4
        self.channel = 4
        self.iter_save=200
        self.epoch=1001
        self.batchsize=32
        self.lr=1e-4
        self.scale = np.array([[120,160],[120,160],[120,160],[120,160]])

args=params()
#blocks
blocks=nn.ModuleDict()
blocks['enc'] = model.Encoder(args.z_dim, args.channel)
blocks['dec'] = model.Decoder(args.z_dim,args.channel,args.z2_dim)
blocks['dag' ]= model.DagLayer(args.z1_dim, args.z1_dim, i = args.inference)
#self.cause = mask.CausalLayer(self.z_dim, self.z1_dim, self.z2_dim)
blocks['attn'] = model.Attention(args.z1_dim)
blocks['mask_z'] = model.MaskLayer(args.z_dim)
blocks['mask_u'] = model.MaskLayer(args.z1_dim,z2_dim=1)
blocks.to(device)

    
# Set prior as fixed parameter attached to Module
z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
z_prior = (z_prior_m,z_prior_v)

#paths
list_attr_path='data/list_attr_celeba.txt'
attr_list=['Smiling','Narrow_Eyes','Mouth_Slightly_Open','Male']
labels_4=get_labels(list_attr_path,attr_list)
imgs_path='data/celeba'

#dataset
train_dataset = dataload_withlabel(imgs_path,labels_4)
train_dataloader = Data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=False,drop_last=True)

#
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
#print(vars(args))
print('CausalVAE')

if not os.path.exists('./figs_vae/'): #判断所在目录下是否有该文件名的文件�?        os.makedirs('./logitdata_{}_{}/train/'.format(sample_num, context_dim))
    os.makedirs('./figs_vae/')

optimizer = torch.optim.Adam([{'params': blocks['enc'].parameters()},{'params': blocks['dec'].parameters()},
            {'params': blocks['dag'].parameters()},{'params': blocks['attn'].parameters()},
            {'params': blocks['mask_z'].parameters()},
            {'params': blocks['mask_u'].parameters()}], lr=args.lr, betas=(0.9, 0.999))

beta = DeterministicWarmup(n=100, t_max=1) # Linear warm-up from 0 to 1 over 50 epoch

#train
for epoch in range(args.epoch):
    for name in blocks:
        blocks[name].train()
    total_loss = 0
    total_rec = 0
    total_kl = 0
    for u, l in train_dataloader:

        optimizer.zero_grad()
        u = u.to(device)
        L, kl, rec, reconstructed_image,_ = negative_elbo_bound(args,blocks,u,l)
        
        dag_param = blocks['dag'].A
        
        h_a = _h_A(dag_param, dag_param.size()[0])
        L = L + 3*h_a + 0.5*h_a*h_a 
   
   
        L.backward()
        optimizer.step()
        total_loss += L.item()
        total_kl += kl.item() 
        total_rec += rec.item() 

        m = len(train_dataset)
        #save_image(u[0], 'figs_vae/reconstructed_image_true_{}.png'.format(epoch), normalize = True) 
        #save_image(reconstructed_image[0], 'figs_vae/reconstructed_image_{}.png'.format(epoch), normalize = True) 
        
    if epoch % 1 == 0:
        print(f"Epoch: {epoch+1}\tL: {total_loss/m:.2f}\tkl: {total_kl/m:.2f}\t rec: {total_rec/m:.2f}")
        print(str(epoch)+' loss:'+str(total_loss/m)+' kl:'+str(total_kl/m)+' rec:'+str(total_rec/m)+' len_dataset:' + str(m))
  
    if epoch %10==0 :
        save_image(u, 'figs_vae/reconstructed_image_true_{}.png'.format(epoch), normalize = True,range=[0,255]) 
        save_image(reconstructed_image, 'figs_vae/reconstructed_image_{}.png'.format(epoch), normalize = True) 
        
    if epoch % args.iter_save == 0:
        torch.save(
                    {
                        "encoder": blocks['enc'].state_dict(),
                        "decoder":blocks['dec'].state_dict(),
                        "attention": blocks['attn'].state_dict(),
                        "daglayer":blocks['dag'].state_dict(),
                        "mask_u": blocks['mask_u'].state_dict(),
                        "mask_z":blocks['mask_z'].state_dict(),
                        "optimizer":optimizer.state_dict(),
                        "args": args
                    },
                    f"checkpoint/{str(epoch).zfill(6)}.pt",
                )