# -*- coding: utf-8 -*-
#
# Distributed under terms of the MIT license.
 
from __future__ import print_function, division
import warnings
import random
import argparse
import numpy as np
from sklearn.cluster import KMeans
#from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
#from torchvision import datasets
from utils_hela import cluster_acc,HelaDataset,USPSDataset
from torch.autograd import Variable
from sklearn.decomposition import PCA
import math
from torch.distributions import normal
torch.manual_seed(100)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib
matplotlib.use('Agg')
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
#%matplotlib inline
from sklearn.decomposition import PCA
from models.imagenet import mobilenetv2
pca = PCA(n_components=5) # The principal components (pca1, pca2, etc.) can be used as features in classification or clustering algorithms.
N = 5000 # limit number of samples for scattering show

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123
#########################################################################

class VAE(nn.Module):
    def __init__(self,
                 n_classes,
                 pretrain_path= 'pretrained/mobilenetv2_1.0-0c6065bc.pth'):        
        super(VAE, self).__init__()
        self.pretrain_path = pretrain_path
        self.mobinetv2 = mobilenetv2()

        self.path=pretrain_path
        self.mobinetv2.load_state_dict(torch.load(self.path))
        #child_counter = 0
        #for child in self.mobinetv2.children():
            #print(" child", child_counter, "is:")
            #print(child)
            #child_counter += 1
        for param in self.mobinetv2.parameters():
            param.requires_grad = True
        num_ftrs = self.mobinetv2.classifier.in_features
        self.mobinetv2.classifier = nn.Linear(num_ftrs, n_classes)       #How to extract only previous layers?     
        #self.mobinetv2.classifier =  nn.Linear(num_ftrs,100) #buffer layer
        #self.pred_layer = nn.Linear(100,n_classes)
        self.adaptinput = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=2,padding=1)             
        torch.nn.init.xavier_uniform(self.adaptinput.weight)  
        self.bn = nn.BatchNorm2d(3)
        print('load pretrained mobinetv2 (1000 --> 10 classes) from', self.path)

    def forward(self, x):
        z = self.adaptinput(x)
        z = self.bn(z) 
        #drop = nn.Dropout(p=0.5)
        #z = drop(z)
        z = self.mobinetv2(z)
        #z = self.pred_layer(z) 
        #z = self.mobinetv2(x)

        return x, z
def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd
 

class IDEC(nn.Module):

    def __init__(self,
                 n_z,
                 n_clusters,
                 alpha=1,
                 pretrain_path= 'vae_hela.pkl'):        
        super(IDEC, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path

        self.vae = VAE(n_z)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self,dataset,supervised_loader,unlabeled_loader, path=''):
        #self.centroids = pretrain_vae(self.vae)
        if path == '':
            self.centroids = pretrain_vae(self.vae,dataset,supervised_loader,unlabeled_loader)
        #load pretrain weights
        self.vae.load_state_dict(torch.load(self.pretrain_path))
        print('load pretrained vae from', path)

    def forward(self, x):

        x_bar, z = self.vae(x)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, q


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()




def pretrain_vae(model,dataset,supervised_loader,unlabeled_loader): #FOR MINIBATCH TRAINING
    '''
    pretrain autoencoder
    '''
    #warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)   
    
    model = VAE(n_classes=args.n_z)
    
    #Supervised learning loss
    criterion = nn.CrossEntropyLoss().cuda()
    ####
    #train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    #_,supervised_loader = split_train_supervised()
    #print(model)
    #model.val()

    indexes = torch.randint(1,862,(15,))
    data = torch.index_select(dataset.x_tr,0,indexes)
    data = torch.Tensor(data).to(device)
    #n_images = 15
    #orig_images = data[:n_images]
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    centroids = None
    '''
    data_show = dataset.x_tr
    y_show = dataset.y_tr
    if (len(y_show) >N):
        data_show = data_show[0:N]
        y_show = y_show[0:N]
    save_image(data_show[:80], 'results_vae/' +'valid_vae_hela.png', nrow=10)
    print('y_show',y_show[:80])    
    data_show = torch.Tensor(data_show).to(device)

    partial_size = 20 #862
    #orig_images = orig_images*0.5+0.5 #inverse normalization    
    '''
    #data = dataset.x_tr
    #data = dataset.data
    #data = torch.Tensor(data).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                #momentum=0.9,
                                #weight_decay=1e-4)    
    model.cuda()
    model.to(device)
    hidden = torch.tensor([]).cuda()
	
    model.eval()    #Caution of Batch Normalization
    hidden = None
    x_bar = None
    # cluster parameter initiate
    data = dataset.x_tr

    supervised_data = [data[i] for i in supervised_idx]

    supervised_data = torch.stack(supervised_data).to(device)
    hidden = partial_vae_forward(model,supervised_data,20)
    _ = kmeans.fit_predict(hidden.detach().cpu().numpy())
    centroids = kmeans.cluster_centers_ 	
	

    ep = 0 
    SN = 5
    Kmeans_labeled = [] 
    predict_labeled = []

    model.train()
    distribution = normal.Normal(0.0,math.sqrt(1))  
    
    #SUPERVISED TRAINING
    for epoch in range(1201): #200
        hidden = torch.tensor([]).cuda()
        true_target= torch.tensor([],dtype=torch.int64).cuda()
        centroids = torch.tensor(centroids).to(device)        
        train_loss = 0
        #for batch_idx, (x, y,_,_, _) in enumerate(train_loader):
        for batch_idx, (x, y,_) in enumerate(supervised_loader):

            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            x_bar, z_batch = model(x)
            hidden = torch.cat((hidden,z_batch))
            true_target = torch.cat((true_target,y))

            supervised_loss = criterion(z_batch,y)
            #loss = supervised_loss
            true_samples = Variable(
                distribution.sample((args.batch_size, args.n_z)),
                requires_grad=False
                ).to(device)

            mmd = compute_mmd(true_samples,z_batch)
            
            q_svdd_centroids = (torch.sum(torch.pow(z_batch.unsqueeze(1) - centroids, 2), 2) )
            q_svdd,_ = torch.min(q_svdd_centroids, dim=1, keepdim=True)
            svdd_loss = torch.sum(q_svdd,0)/x.shape[0] 
            total_d = torch.sum(q_svdd_centroids, dim=1, keepdim=True)
            d_xi_cj = total_d - q_svdd
            d_xi_cj_loss = 1.0/torch.sum(d_xi_cj,0)/x.shape[0]
            loss = 1.0*supervised_loss + mmd #+ args.gamma*svdd_loss + args.gamma2*d_xi_cj_loss #0.1 svdd -> 0.11 acc; 0*svdd ->0.88 acc
                    
            
            
            '''
            q_svdd_centroids = (torch.sum(torch.pow(hidden.unsqueeze(1) - torch.tensor(centroids).to(device), 2), 2) )
            q_svdd,_ = torch.min(q_svdd_centroids, dim=1, keepdim=True)
            svdd_loss = torch.sum(q_svdd,0)/x.shape[0] 
            #total_d = torch.sum(q_svdd_centroids, dim=1, keepdim=True)
            top2_vals = torch.topk(q_svdd_centroids, k=2,largest=False, dim=1)[0]
            #print('total_d shape', total_d.shape)
            #d_xi_cj = total_d - q_svdd
            d_xi_cj = top2_vals[:,1]
            #print('d_xi_cj shape', d_xi_cj.shape)
            #print('q_svdd shape', q_svdd.shape)
            d_xi_cj_loss = 1.0/(torch.sum(d_xi_cj,0)/x.shape[0])
            #print('d_xi_cj_loss shape', d_xi_cj_loss.shape)
            #loss = 1.0*supervised_loss + args.gamma*svdd_loss + args.gamma2*d_xi_cj_loss #0.1 svdd -> 0.11 acc; 0*svdd ->0.88 acc
            
            
            
            loss = 1.0*supervised_loss + gamma_1*svdd_loss + gamma_2*d_xi_cj_loss #0.1 svdd -> 0.11 acc; 0*svdd ->0.88 acc        
            '''
            #loss.backward(retain_graph=True) #Consume memory   Be careful with RNN structure, e.g z_batch -> hidden of concat -> need retain graph = true, even if loss = loss1+loss2         
            
            
            loss.backward()
            train_loss += loss.item() 
            optimizer.step()
            torch.cuda.empty_cache()




        
        if epoch%SN==0:        
            z_show = hidden
            y_show = true_target.detach().cpu().numpy()      
            #fashion_tsne_hidden_without_pca = TSNE(random_state=RS).fit_transform(z_show.detach().cpu())
            #fashion_scatter(fashion_tsne_hidden_without_pca, y_show,epoch,'pretrain_vae')
            
            y_pred = kmeans.fit_predict(z_show.detach().cpu().numpy())
            acc = cluster_acc(y_show, y_pred)
    
            print('KMeans Z: Labeled Acc {:.4f}'.format(acc))
            #if epoch %100==0:
                #centroids = kmeans.cluster_centers_
            y_pred_cls = hidden.detach().cpu().numpy().argmax(1)
    
            acc_cls = cluster_acc(y_show, y_pred_cls)
            print('Classifier : Acc {:.4f}'.format(acc_cls))
            
            
            #Update centroids    
            centroids = kmeans.cluster_centers_            


            #Plot training curve
            predict_labeled.append(acc)
            Kmeans_labeled.append(acc_cls)
            ep = ep+SN            

        print("epoch {} loss={:.4f}".format(epoch, train_loss / (batch_idx + 1)))   
        
    torch.cuda.empty_cache() 

    t = str(random.randint(1,1000))
    plt.plot(range(0,ep,SN),predict_labeled)
    #plt.plot(range(ep),[0.1,0.2,0.3])
    plt.savefig('./results_mobiv2/Predict_Labeled_ACC_'+ t +'.pdf')
    plt.close()    
    plt.plot(range(0,ep,SN),Kmeans_labeled)
    #plt.plot(range(ep),[0.1,0.2,0.3])
    plt.savefig('./results_mobiv2/Kmeans_Labeled_ACC_' + t + '.pdf') 
    plt.close()




    model.eval()    #Caution of Batch Normalization
    hidden = None

    # cluster parameter initiate
    data = dataset.x_tr
    y = dataset.y_tr
    
    unsupervised_data = [data[i] for i in unsupervised_idx]
    unsupervised_data = torch.stack(unsupervised_data).to(device)
    unsupervised_y = [y[i] for i in unsupervised_idx]
    unsupervised_y = np.array(unsupervised_y)
    #unsupervised_y = torch.stack(unsupervised_y) 
    #data = torch.Tensor(data).to(device)
    
    #print('unsupervised_data_size0', unsupervised_data.shape)
    #print('y_size0', y.shape)                            
    #print('unsupervised_y_size0', unsupervised_y.shape)   
    #_,hidden = model.vae(data)
    hidden = partial_vae_forward(model,unsupervised_data,20)
    #hidden = hidden[:,:2]
    #print('hidden_size', hidden.shape)

    unsupervised_y_pred = hidden.detach().cpu().numpy().argmax(1)
    #print('unsupervised_y_pred_size0', unsupervised_y_pred.shape)      
    unsupervised_acc = cluster_acc(unsupervised_y, unsupervised_y_pred)
    print('After trained : unsupervised_ Acc {:.4f}'.format(unsupervised_acc))   


    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    unsupervised_y_pred_kmeans = kmeans.fit_predict(hidden.data.cpu().numpy())      
    unsupervised_acc = cluster_acc(unsupervised_y, unsupervised_y_pred_kmeans)
    print('After trained unsupervised_ KMeans : Acc {:.4f}'.format(unsupervised_acc)) 
    
    z_show = hidden
    y_show = unsupervised_y  
    fashion_tsne_hidden_without_pca = TSNE(random_state=RS).fit_transform(z_show.detach().cpu())
    fashion_scatter(fashion_tsne_hidden_without_pca, y_show,0,'post_trained_vae_'+t)  
    
    
    torch.save(model.state_dict(), args.pretrain_path)
    print("model saved to {}.".format(args.pretrain_path))
    torch.cuda.empty_cache() 
    return centroids
    
def partial_vae_forward(model,data,partial_size):          #FOR MINIBATCH REFERENCE
    model.eval() #Caution of Batch Normalization
    hidden = torch.tensor([])#.to(device)
    #data_batch = data.size(0)
    data_batch = data.shape[0]
    m = int(data_batch/partial_size)
    n = data_batch%partial_size
    for i in range(m):
        partial_data = data[i*partial_size:(i+1)*partial_size]
        #partial_data = partial_data.to(device)
        x_bar_batch, hidden_batch = model(partial_data)
        hidden = torch.cat((hidden, hidden_batch.data.cpu()))
    if n>0:    
        partial_data = data[m*partial_size:]
        #partial_data = partial_data.to(device)
        x_bar_batch, hidden_batch = model(partial_data)
        hidden = torch.cat((hidden, hidden_batch.data.cpu()))
        #torch.cuda.empty_cache()    
    return hidden

def partial_vae(model,data,partial_size):          #FOR MINIBATCH REFERENCE
    model.eval()    #Caution of Batch Normalization
    hidden = torch.tensor([])#.to(device)
    #data_batch = data.size(0)
    data_batch = data.shape[0]
    
    m = int(data_batch/partial_size)
    n = data_batch%partial_size    
    for i in range(m):
        partial_data = data[i*partial_size:(i+1)*partial_size]
        #partial_data = partial_data.to(device)
        x_bar_batch, hidden_batch = model.vae(partial_data) #vae from idec model
        hidden = torch.cat((hidden, hidden_batch.data.cpu()))
        #torch.cuda.empty_cache()
    if n>0:    
        partial_data = data[m*partial_size:]
        #partial_data = partial_data.to(device)
        x_bar_batch, hidden_batch = model(partial_data)
        hidden = torch.cat((hidden, hidden_batch.data.cpu()))        
    return hidden

def partial_model(model,data,partial_size):  #FOR MINIBATCH REFERENCE
    model.eval()    #Caution of Batch Normalization
    x_bar = torch.tensor([])#.to(device)
    tmp_q = torch.tensor([])#.to(device)
    m = int(data.size(0)/partial_size)
    n = data.size(0)%partial_size    
    for i in range(m):
        partial_data = data[i*partial_size:(i+1)*partial_size]
        partial_data = partial_data.to(device)
        #print('partial_data shape', partial_data.shape)
        x_bar_batch, tmp_q_batch = model(partial_data)
        x_bar = torch.cat((x_bar, x_bar_batch.data.cpu()))
        tmp_q = torch.cat((tmp_q, tmp_q_batch.data.cpu())) 
    if n>0:
        partial_data = data[m*partial_size:]
        partial_data = partial_data.to(device)
        #print('partial_data shape', partial_data.shape)
        x_bar_batch, tmp_q_batch = model(partial_data)
        x_bar = torch.cat((x_bar, x_bar_batch.data.cpu()))
        tmp_q = torch.cat((tmp_q, tmp_q_batch.data.cpu())) 
        torch.cuda.empty_cache()
    #print('tmp_q shape', tmp_q.shape)        
    return x_bar, tmp_q


def indices_shuffling(dataset): 
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=5,random_state=random.randint(1,1000),shuffle=True) #corresponding to split  ratio of 0.8
    ratio_samples_number =[]
    train_dataset = dataset
    X = dataset.x_tr
    y = dataset.y_tr
    skf.get_n_splits(X, y)    
    #random_seed=100
    num_train = len(train_dataset)
    #indices = list(range(num_train))
    indices = list()
    for train_index, unsupervised_index in skf.split(X, y):
        indices.extend(unsupervised_index) #indices +=unsupervised_index
        ratio_samples_number.append(len(unsupervised_index))
    #split = int(np.floor(supervised_size * num_train))
    #np.random.seed(random_seed)
    #np.random.shuffle(indices)
    #stratifiedKfold 10 times to goet group of indexes
    #concatenateToList    
    #print('len indices',len(indices))
    print('indices=',indices) 
    return indices,num_train,ratio_samples_number
def split_train_supervised(train_dataset,indices,num_train,ratio_samples_number,ratio,k):
    unsupervised_start = 0
    unsupervised_size = ratio_samples_number[k]

    for i in range(k):
        unsupervised_start += ratio_samples_number[i]   

    unsupervised_end = unsupervised_start+unsupervised_size
    unsupervised_idx = indices[unsupervised_start: unsupervised_end+1] # +1 due to slicing op
    print('shape unsupervised_idx',len(unsupervised_idx))
    y = train_dataset.y_tr
    values, counts = np.unique(y, return_counts=True)
    print('dataset structures',[values,counts])
    print('unsupervised_idx=',sorted(unsupervised_idx))    
    print('unsupervised_y =',sorted(y[unsupervised_idx]))

    supervised_idx = indices[ : unsupervised_start] + indices[unsupervised_end+1:]
    #unlabeled_idx, supervised_idx = indices[split:], indices[:split]
    unsupervised_sampler = SubsetRandomSampler(unsupervised_idx)
    supervised_sampler = SubsetRandomSampler(supervised_idx)

    unlabeled_loader = torch.utils.data.DataLoader(
        train_dataset,sampler=unsupervised_sampler,
        batch_size=args.batch_size,
        num_workers=0, pin_memory=True)
    
    supervised_loader = torch.utils.data.DataLoader(
        train_dataset,sampler=supervised_sampler,
        batch_size=args.batch_size, 
        num_workers=0, pin_memory=True)     
    
    return unlabeled_loader, supervised_loader,supervised_idx,unsupervised_idx

def train_idec(dataset,supervised_loader,unlabeled_loader):

    #warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)     

    model = IDEC(
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        alpha=1.0,
        pretrain_path=args.pretrain_path).to(device)

    #model.pretrain('vae_hela.pkl')
    model.pretrain(dataset,supervised_loader,unlabeled_loader,'')
    centroids = model.centroids
    #print('centroids shape',centroids.shape)
    #model.load_state_dict(torch.load('data/idec_vae_mnist_40.pkl')) #Load previous training, due to limited  CUDA memory
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)    #SHOULD KEEP IT FALSE Shuffle to match the q distributions 

    '''
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)
    '''
    optimizer = Adam(model.parameters(), lr=0.001)

    # cluster parameter initiate

    model.eval()    #Caution of Batch Normalization
    hidden = None
    x_bar = None
    # cluster parameter initiate
    data = dataset.x_tr
    y = dataset.y_tr
    
    unsupervised_data = [data[i] for i in unsupervised_idx]
    unsupervised_data = torch.stack(unsupervised_data).to(device)
    unsupervised_y = [y[i] for i in unsupervised_idx]
    unsupervised_y = np.array(unsupervised_y)
    #unsupervised_y = torch.stack(unsupervised_y) 
    #data = torch.Tensor(data).to(device)
    
    print('unsupervised_data_size0', unsupervised_data.shape)
    print('y_size0', y.shape)                            
    print('unsupervised_y_size0', unsupervised_y.shape)   
    #_,hidden = model.vae(data)
    hidden = partial_vae(model,unsupervised_data,20)
    #hidden = hidden[:,:2]
    print('hidden_size', hidden.shape)

    unsupervised_y_pred = hidden.detach().cpu().numpy().argmax(1)
    print('unsupervised_y_pred_size0', unsupervised_y_pred.shape)      
    unsupervised_acc = cluster_acc(unsupervised_y, unsupervised_y_pred)
    print('Beginning : unsupervised_ Acc {:.4f}'.format(unsupervised_acc))   


    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    unsupervised_y_pred_kmeans = kmeans.fit_predict(hidden.data.cpu().numpy())      
    unsupervised_acc = cluster_acc(unsupervised_y, unsupervised_y_pred_kmeans)
    print('Beginning unsupervised_ KMeans : Acc {:.4f}'.format(unsupervised_acc)) 
    
    z_show = hidden
    y_show = unsupervised_y  
    fashion_tsne_hidden_without_pca = TSNE(random_state=RS).fit_transform(z_show.detach().cpu())
    fashion_scatter(fashion_tsne_hidden_without_pca, y_show,0,'post_trained_vae')   
    
    y_pred_last = unsupervised_y_pred
    model.cluster_layer.data = torch.tensor(centroids).to(device) #NOTE: what is the data of the cluster layer?
    #model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
  

  
    for epoch in range(1): #.0
        tmpq = torch.tensor([]).to(device)
  
        true_target= torch.tensor([],dtype=torch.int64).to(device)
        tmpq_u = torch.tensor([]).to(device)
  
        true_target_u= torch.tensor([],dtype=torch.int64).to(device)        

        if epoch % args.update_interval == 0:
            model.eval() #Because of Batch Normalization layer, using statistics of whole dataset
            for batch_idx, (x, y,_) in enumerate(train_loader):

                x = x.to(device)
                y = y.to(device)
                _, tmp_q = model(x)
                # update target distribution p
                tmp_q = tmp_q.data
                tmpq = torch.cat((tmpq,tmp_q))
                true_target = torch.cat((true_target,y))
            p = target_distribution(tmpq)
            #print('tmpq',tmpq.shape)            
            #print('p shape',p.shape)
            #print('tmpq = ',tmpq[:5,:])            
            #print('p shape = ',p[:5,:])            
                
            p = p.to(device)
            tmpq = tmpq.to(device)
            y_true = true_target.detach().cpu().numpy()    

            # evaluate clustering performance
            y_pred = tmpq.cpu().numpy().argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                    np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            #print('true target shape',true_target.shape) 
            #print('y pred shape',y_pred.shape) 
            acc = cluster_acc(y_true, y_pred)

            print('Iter {}'.format(epoch), ':mixed Acc {:.4f}'.format(acc))
            
            for batch_idx_u, (x_u, y_u,_) in enumerate(unlabeled_loader):

                x_u = x_u.to(device)
                y_u = y_u.to(device)
                _, tmp_q_u = model(x_u)
                # update target distribution p
                tmp_q_u = tmp_q_u.data
                tmpq_u = torch.cat((tmpq_u,tmp_q_u))
                true_target_u = torch.cat((true_target_u,y_u))

            y_true_u = true_target_u.detach().cpu().numpy()    

            # evaluate clustering performance
            y_pred_u = tmpq_u.cpu().numpy().argmax(1)
 
            acc_u = cluster_acc(y_true_u, y_pred_u)

            print('Iter {}'.format(epoch), ':Unalabeled Acc {:.4f}'.format(acc_u))
            
            

            if epoch > 0 and delta_label < args.tol:
                print('delta_label {:.4f}'.format(delta_label), '< tol',
                      args.tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        
        for batch_idx, (x_batch, y_batch,idx) in enumerate(train_loader):
        #for batch_idx, (x_batch, y_batch,_,_,idx) in enumerate(train_loader):

            
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            idx = idx.type(torch.LongTensor).to(device)
            model.eval() #Because of Batch Normalization layer, using statistics of whole dataset
            #_, hidden,mu,logvar = model.vae(x_batch)
            x_bar, q = model(x_batch)
            model.train()            
            #loss = calcul_loss(x_bar, x_batch, mu, logvar)
            #loss = loss/(x_batch.shape[0]*784)
            #reconstr_loss = F.mse_loss(x_bar, x_batch)
            #BCE = torch.nn.functional.binary_cross_entropy(x_bar, x_batch.view(-1, 1,28,28), reduction='mean')
            kl_loss = F.kl_div(q.log(), p[idx]) # PYTORCH implementation: the `input` given is expected to contain *log-probabilities*
            #loss = reconstr_loss + args.gamma * kl_loss 
            loss = 0.01*kl_loss #+BCE
            #loss = args.gamma *reconstr_loss +  kl_loss 
            #print('reconstr_loss: {:.4f}'.format(reconstr_loss))
            #print('kl_loss: {:.4f}'.format(kl_loss))
            #print('loss: {:.4f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            
            
        if epoch%2==0:
            model.eval()            
            hidden = partial_vae(model,data.to(device),20)            
            z_show = hidden
            y_show = y_true     
            fashion_tsne_hidden_without_pca = TSNE(random_state=RS).fit_transform(z_show.detach().cpu())
            fashion_scatter(fashion_tsne_hidden_without_pca, y_show,epoch,'train_cluster')
            torch.cuda.empty_cache()  
        torch.cuda.empty_cache()    
        #x_bar, hidden,mu,logvar = model.vae(data)
    #comparison = torch.cat([data[:n], x_bar.view(x_bar.shape[0], 1, 28, 28)[:n]])
    #save_image(comparison.cpu(), 'data/results_vae/' +'valid_vae_'+ str(epoch) +  '.png', nrow=n)    
    #torch.save(model.state_dict(), 'data/idec_vae_stl10.pkl')
    #print("model saved to {}.".format('data/idec_vae_stl10.pkl'))
    '''    
    data = dataset.x_te
    y = dataset.y_te
    data = torch.Tensor(data).to(device)    
    _, tmp_q = model(data)
    # evaluate clustering performance
    y_pred = tmp_q.detach().cpu().numpy().argmax(1)
    acc = cluster_acc(y, y_pred)
    nmi = nmi_score(y, y_pred)
    ari = ari_score(y, y_pred)
    print('Testing: Acc {:.4f}'.format(acc),
             ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))
    '''


def fashion_scatter(x, colors,idx,message):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    #sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # show the text for digit corresponding to the true label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0) #true labels 
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    #plt.savefig('./results_vae/scatter_'+ str(idx) + '.png', bbox_inches='tight')
    plt.savefig('./results_mobiv2/scatter_'+ str(idx) +'_'+message+ '.pdf', bbox_inches='tight')
    plt.close()
    return f, ax, sc, txts



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--dataset', type=str, default='hela')
    parser.add_argument('--ratio', default=8, type=int) #corresponding to 1 * 10%
    parser.add_argument('--pretrain_path', type=str, default='vae_hela')
    parser.add_argument(
        '--gamma',
        default=0.001, #0.1
        type=float,
        help='coefficient of clustering loss')
    parser.add_argument(
        '--gamma2',
        default=0.1, #0.1
        type=float,
        help='coefficient 2 of clustering loss')    
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.dataset == 'mnist':
        args.pretrain_path = 'data/vae_mnist.pkl'
        args.n_clusters = 10
        args.n_input = 28*28
        dataset = MnistDataset()
    if args.dataset == 'hela':
        args.pretrain_path = 'vae_hela.pkl'
        args.n_clusters = 10
        args.n_input = 448*448
        dataset = HelaDataset()
        shuffled_indices, num_train,ratio_samples_number = indices_shuffling(dataset)       
        supervised_loader,unlabeled_loader,unsupervised_idx,supervised_idx = split_train_supervised(dataset,shuffled_indices, num_train,ratio_samples_number,args.ratio,k=0)
        train_idec(dataset,supervised_loader,unlabeled_loader)
        supervised_loader,unlabeled_loader,unsupervised_idx,supervised_idx = split_train_supervised(dataset,shuffled_indices, num_train,ratio_samples_number,args.ratio,k=1)
        train_idec(dataset,supervised_loader,unlabeled_loader)
        supervised_loader,unlabeled_loader,unsupervised_idx,supervised_idx = split_train_supervised(dataset,shuffled_indices, num_train,ratio_samples_number,args.ratio,k=2)
        train_idec(dataset,supervised_loader,unlabeled_loader)
        supervised_loader,unlabeled_loader,unsupervised_idx,supervised_idx = split_train_supervised(dataset,shuffled_indices, num_train,ratio_samples_number,args.ratio,k=3)
        train_idec(dataset,supervised_loader,unlabeled_loader)
        supervised_loader,unlabeled_loader,unsupervised_idx,supervised_idx = split_train_supervised(dataset,shuffled_indices, num_train,ratio_samples_number,args.ratio,k=4)
        train_idec(dataset,supervised_loader,unlabeled_loader)
       
           
    if args.dataset == 'usps':
        args.pretrain_path = 'data/vae_usps.pkl'
        args.n_clusters = 10
        args.n_input = 16*16
        dataset = USPSDataset()
    if args.dataset == 'stl10':
        args.pretrain_path = 'data/vae_stl10.pkl'
        args.n_clusters = 10
        args.n_input = 96*96
        dataset = STL10Dataset()
    if args.dataset == 'cifar10':
        args.pretrain_path = 'data/vae_cifar10.pkl'
        args.n_clusters = 10
        args.n_input = 32*32
        dataset = CIFAR10Dataset()        
    #print(args)
    #pretrain_vae(dataset,supervised_loader,unlabeled_loader)
    #train_idec(dataset,supervised_loader,unlabeled_loader)
    #test_idec()
