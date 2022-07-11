
from __future__ import print_function, division
import warnings
import random
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils_hela import cluster_acc,HelaDataset
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
                 saved_model= 'pretrained/mobilenetv2_1.0-0c6065bc.pth'):        
        super(VAE, self).__init__()
        
        self.mobinetv2 = mobilenetv2()

        self.path=saved_model
        num_ftrs = self.mobinetv2.classifier.in_features

        self.mobinetv2.classifier = nn.Linear(num_ftrs, n_classes)       
            
        print('load pretrained mobinetv2 (1000 -->100 --> 10 classes) from', self.path)

    def forward(self, x):

        z = self.mobinetv2(x)

        return x, z



def train_organet(dataset,supervised_loader,unlabeled_loader): #FOR MINIBATCH TRAINING
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

    indexes = torch.randint(1,862,(15,))
    data = torch.index_select(dataset.x_tr,0,indexes)
    data = torch.Tensor(data).to(device)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    centroids = None

    optimizer = Adam(model.parameters(), lr=args.lr)
 
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
    SN = 10
    predict_labeled = []
  

    model.train()

    
    #SUPERVISED TRAINING
    for epoch in range(1501): #200
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
            
            q_svdd_centroids = (torch.sum(torch.pow(z_batch.unsqueeze(1) - centroids, 2), 2) )
            q_svdd,_ = torch.min(q_svdd_centroids, dim=1, keepdim=True)
            svdd_loss = torch.sum(q_svdd,0)/x.shape[0] 
            total_d = torch.sum(q_svdd_centroids, dim=1, keepdim=True)
            d_xi_cj = total_d - q_svdd
            d_xi_cj_loss = 1.0/torch.sum(d_xi_cj,0)/x.shape[0]
            loss = 1.0*supervised_loss + args.gamma*svdd_loss + args.gamma2*d_xi_cj_loss
    
                         
            loss.backward()
            train_loss += loss.item() 
            optimizer.step()
            torch.cuda.empty_cache()


        
        if epoch%SN==0:        
            z_show = hidden
            y_show = true_target.detach().cpu().numpy()      
            
            y_pred = kmeans.fit_predict(z_show.detach().cpu().numpy())
            acc = cluster_acc(y_show, y_pred)
    
            print('KMeans Z: Labeled Acc {:.4f}'.format(acc))          
            
            #Update centroids    
            centroids = kmeans.cluster_centers_            


            #Plot training curve
            predict_labeled.append(acc)

            ep = ep+SN            

        print("epoch {} loss={:.4f}".format(epoch, train_loss / (batch_idx + 1)))   
        
    torch.cuda.empty_cache() 

    t = str(random.randint(1,1000))
    plt.plot(range(0,ep,SN),predict_labeled)
    #plt.plot(range(ep),[0.1,0.2,0.3])
    plt.savefig('./results/Predict_Labeled_ACC_'+ t +'.pdf')
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

    hidden = partial_vae_forward(model,unsupervised_data,20)


    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    unsupervised_y_pred_kmeans = kmeans.fit_predict(hidden.data.cpu().numpy())      
    unsupervised_acc = cluster_acc(unsupervised_y, unsupervised_y_pred_kmeans)
    print('After trained unsupervised_ KMeans : Acc {:.4f}'.format(unsupervised_acc)) 
    
    z_show = hidden
    y_show = unsupervised_y  
    fashion_tsne_hidden_without_pca = TSNE(random_state=RS).fit_transform(z_show.detach().cpu())
    fashion_scatter(fashion_tsne_hidden_without_pca, y_show,0,'post_trained_vae')  
    
    
    torch.save(model.state_dict(), args.saved_model)
    print("model saved to {}.".format(args.saved_model))
    torch.cuda.empty_cache() 
    return centroids
    
def partial_vae_forward(model,data,partial_size):         
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



def indices_shuffling(dataset): 
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=10,random_state=random.randint(1,1000),shuffle=True) #corresponding to split  ratio of 0.9
    ratio_samples_number =[]
    train_dataset = dataset
    X = dataset.x_tr
    y = dataset.y_tr
    skf.get_n_splits(X, y)    

    num_train = len(train_dataset)

    indices = list()
    for train_index, supervised_index in skf.split(X, y):
        indices.extend(supervised_index) #indices +=supervised_index
        ratio_samples_number.append(len(supervised_index))

    return indices,num_train,ratio_samples_number
def split_train_supervised(train_dataset,indices,num_train,ratio_samples_number,ratio):
    supervised_size = 0
    for i in range(ratio):
        supervised_size += ratio_samples_number[i]   


    supervised_idx = indices[: supervised_size+1]
    y = train_dataset.y_tr
    values, counts = np.unique(y, return_counts=True)
    print('dataset structures',[values,counts])

    unsupervised_idx = indices[supervised_size+1:]
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
    plt.savefig('./results/scatter_'+ str(idx) +'_'+message+ '.pdf', bbox_inches='tight')
    plt.close()
    return f, ax, sc, txts



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--dataset', type=str, default='hela')
    parser.add_argument('--ratio', default=8, type=int) #corresponding to 1 * 10%
    parser.add_argument('--saved_model', type=str, default='vae_hela')
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

    if args.dataset == 'hela':
        args.saved_model = 'vae_hela.pkl'
        args.n_clusters = 10
        args.n_input = 224*224
        dataset = HelaDataset()
        shuffled_indices, num_train,ratio_samples_number = indices_shuffling(dataset)       
        unlabeled_loader,supervised_loader,supervised_idx,unsupervised_idx = split_train_supervised(dataset,shuffled_indices, num_train,ratio_samples_number,args.ratio)
            
      
    #print(args)
    train_organet(dataset,supervised_loader,unlabeled_loader)

