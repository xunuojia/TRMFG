# coding=utf-8


import torch
from torch import nn
import dgl

import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
import dgl.function as fn
import math

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)





class HERE_Last(torch.nn.Module):
    def __init__(self,q_e,t_e,in_size,out_size):
        super(HERE_Last, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        
        self.q_emb = torch.nn.Parameter(torch.FloatTensor(q_e), requires_grad = False)
        # nn.init.xavier_uniform_(self.q_emb)

        self.t_emb = torch.nn.Parameter(torch.FloatTensor(t_e), requires_grad = False)
        
        # self.Embed = nn.Embedding(vocab_length,hidden_states)
        self.mlp = nn.Linear(self.in_size,self.out_size)#need  act relu
        self.mlp_linear=nn.Linear(self.out_size,1)#no act

        self.conv=nn.Conv2d(self.in_size,self.out_size,kernel_size=2,stride=1,padding=(0,0))
        self.conv1=nn.Conv2d(self.in_size,self.out_size,kernel_size=3,stride=1,padding=(1,0))
        self.conv2=nn.Conv2d(self.in_size,self.out_size,kernel_size=4,stride=1,padding=(1,0))

        # self.mlp=nn.Dense(units=hidden_states,flatten=True,activation='relu')
        # self.mlp_linear=nn.Dense(units=1,flatten=True)
        # self.conv=nn.Conv2d(self.in_size,self.out_size,kernel_size=(2,self.out_size),stride=(1,self.out_size),padding=(0,0))
        # self.conv1=nn.Conv2d(self.in_size,self.out_size,kernel_size=(3,self.out_size),stride=(1,self.out_size),padding=(1,0))
        # self.conv2=nn.Conv2d(self.in_size,self.out_size,kernel_size=(4,self.out_size),stride=(1,self.out_size),padding=(1,0))


        self.att_des=nn.Linear(self.in_size,self.out_size)#need  act relu
        self.att_des_final=nn.Linear(self.out_size,1)#no act


        self.gatetag=nn.Linear(self.in_size,self.out_size)#need  act sigmoid
        self.gatedes=nn.Linear(self.in_size,self.out_size)#need  act sigmoid

        self.attention=nn.Linear(self.in_size,self.out_size)#need  act sigmoid


        self.agg=nn.Linear(self.in_size,self.out_size)#need  act relu
        self.agg_final=nn.Linear(self.out_size,1)#no act


        self.trans_map=nn.Linear(self.in_size*4,self.out_size)#need  act relu
        self.trans_ques=nn.Linear(self.in_size,self.out_size)#need  act relu

        self.layer=nn.LayerNorm(self.q_emb.shape)
        self.p_emb = torch.zeros((self.q_emb.shape[0],self.out_size))
        self.pe = None
        self.device = None
        

        
        
    def forward(self,):
        # question Encode layer
        
        def pos(p_emb,ques_length,hidden_states):
            
            
            position=torch.arange(0,ques_length).reshape((ques_length,1))
            
            div_term=torch.exp(torch.arange(0,hidden_states,2)*-(math.log(10000)/hidden_states))
            
            p_emb[:,0::2]=torch.sin(position*div_term)
            p_emb[:,1::2]=torch.cos(position*div_term)
            pe = p_emb
            
            return pe


        

        
        question =  self.q_emb
        self.device = question.device

        # question =  self.Embed(question)
        # question=ques_mask*question


        ques_temp=question
        question_2 = F.relu(self.mlp(ques_temp))
        question_3 = F.relu(self.mlp(ques_temp))
        question_4 = F.relu(self.mlp(ques_temp))

        question_all = torch.concat([question,question_2,question_3,question_4],dim=1)
        question_all = F.relu(self.trans_map(question_all))

        self.pe=pos(self.p_emb,self.q_emb.shape[0],self.out_size)
        self.pe = self.pe.to(self.device)
        


        question_all=question_all+self.pe
        question_all=F.relu(self.trans_ques(question_all))



        
        question_all=self.layer(question_all+question)



        #topic encoder layer

        topicall = self.t_emb
        gatetag=F.sigmoid(self.gatetag(topicall))

        tag_all=gatetag*topicall
        return question_all,tag_all