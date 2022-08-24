
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

class OrgaNet(nn.Module):
    def __init__(self,
                 n_classes,
                 saved_model= 'pretrained/mobilenetv2_1.0-0c6065bc.pth'):        
        super(OrgaNet, self).__init__()
        
        self.mobinetv2 = mobilenetv2()

        self.path=saved_model
        num_ftrs = self.mobinetv2.classifier.in_features

        self.mobinetv2.classifier = nn.Linear(num_ftrs, n_classes)       
            
        print('load pretrained mobinetv2 (1000 --> 10 classes) from', self.path)

    def forward(self, x):
        z = self.mobinetv2(x)

        return z

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


def train_organet(dataset,training_loader,testing_loader): #FOR MINIBATCH TRAINING

    #warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)   
    
    model = OrgaNet(n_classes=args.n_z)
    
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

    # cluster parameter initiate
    data = dataset.x_tr

    training_data = [data[i] for i in training_idx]

    training_data = torch.stack(training_data).to(device)
    hidden = partial_OrgaNet_forward(model,training_data,20)
    _ = kmeans.fit_predict(hidden.detach().cpu().numpy())
    centroids = kmeans.cluster_centers_ 	
	
    ep = 0 
    SN = 10
    predict_labeled = []
  

    model.train()
    distribution = normal.Normal(0.0,math.sqrt(1))  
    
    #SUPERVISED TRAINING
    for epoch in range(301): #200
        hidden = torch.tensor([]).cuda()
        true_target= torch.tensor([],dtype=torch.int64).cuda()
        centroids = torch.tensor(centroids).to(device)        
        train_loss = 0
        #for batch_idx, (x, y,_,_, _) in enumerate(train_loader):
        for batch_idx, (x, y,_) in enumerate(training_loader):

            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            z_batch = model(x)
            hidden = torch.cat((hidden,z_batch))
            true_target = torch.cat((true_target,y))

            supervised_loss = criterion(z_batch,y)
            #loss = supervised_loss
            true_samples = Variable(
                distribution.sample((args.batch_size, args.n_z)),
                requires_grad=False
                ).to(device)

            mmd = compute_mmd(true_samples,z_batch)
            
            loss = supervised_loss + 1.0*mmd 
    
                         
            loss.backward()
            train_loss += loss.item() 
            optimizer.step()
            torch.cuda.empty_cache()


        
        if epoch%SN==0:        
            z_show = hidden
            y_show = true_target.detach().cpu().numpy()      
            
            y_pred = kmeans.fit_predict(z_show.detach().cpu().numpy())
            acc = cluster_acc(y_show, y_pred)
    
            print('Training Acc. by clustering {:.4f}'.format(acc))          
            
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
    
    testing_data = [data[i] for i in testing_idx]
    testing_data = torch.stack(testing_data).to(device)
    testing_y = [y[i] for i in testing_idx]
    testing_y = np.array(testing_y)

    hidden = partial_OrgaNet_forward(model,testing_data,20)


    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    testing_y_pred_kmeans = kmeans.fit_predict(hidden.data.cpu().numpy())      
    testing_acc = cluster_acc(testing_y, testing_y_pred_kmeans)
    print('After trained : Acc {:.4f}'.format(testing_acc)) 
    
    z_show = hidden
    y_show = testing_y  
    fashion_tsne_hidden_without_pca = TSNE(random_state=RS).fit_transform(z_show.detach().cpu())
    fashion_scatter(fashion_tsne_hidden_without_pca, y_show,0,'post_trained_OrgaNet')  
    
    
    torch.save(model.state_dict(), args.saved_model)
    print("model saved to {}.".format(args.saved_model))
    torch.cuda.empty_cache() 
    return centroids
    
def partial_OrgaNet_forward(model,data,partial_size):         
    model.eval() #Caution of Batch Normalization
    hidden = torch.tensor([])#.to(device)
    data_batch = data.shape[0]
    m = int(data_batch/partial_size)
    n = data_batch%partial_size
    for i in range(m):
        partial_data = data[i*partial_size:(i+1)*partial_size]
        #partial_data = partial_data.to(device)
        hidden_batch = model(partial_data)
        hidden = torch.cat((hidden, hidden_batch.data.cpu()))
    if n>0:    
        partial_data = data[m*partial_size:]
        #partial_data = partial_data.to(device)
        hidden_batch = model(partial_data)
        hidden = torch.cat((hidden, hidden_batch.data.cpu()))
        #torch.cuda.empty_cache()    
    return hidden

def indices_shuffling(dataset): 
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=10,random_state=random.randint(1,1000),shuffle=True) #corresponding to testing  ratio of 0.1
    ratio_samples_number =[]
    train_dataset = dataset
    X = dataset.x_tr
    y = dataset.y_tr
    skf.get_n_splits(X, y)    

    num_train = len(train_dataset)

    indices = list()
    for _ , random_testing_index in skf.split(X, y):
        indices.extend(random_testing_index) 
        ratio_samples_number.append(len(random_testing_index))

    return indices,num_train,ratio_samples_number
def split_train_test(train_dataset,indices,num_train,ratio_samples_number,ratio):
    training_set_size = 0
    for i in range(ratio):
        training_set_size += ratio_samples_number[i]   


    training_idx = indices[: training_set_size+1]
    y = train_dataset.y_tr
    values, counts = np.unique(y, return_counts=True)
    print('dataset structures',[values,counts])

    testing_idx = indices[training_set_size+1:]
    testing_sampler = SubsetRandomSampler(testing_idx)
    training_sampler = SubsetRandomSampler(training_idx)

    testing_loader = torch.utils.data.DataLoader(
        train_dataset,sampler=testing_sampler,
        batch_size=args.batch_size,
        num_workers=0, pin_memory=True)
    
    training_loader = torch.utils.data.DataLoader(
        train_dataset,sampler=training_sampler,
        batch_size=args.batch_size, 
        num_workers=0, pin_memory=True)     
    
    return testing_loader, training_loader,training_idx,testing_idx




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
    parser.add_argument('--ratio', default=8, type=int) #corresponding to 8 * 10%
    parser.add_argument('--saved_model', type=str, default='OrgaNet_hela')
    
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.dataset == 'hela':
        args.saved_model = 'OrgaNet_hela.pkl'
        args.n_clusters = 10
        args.n_input = 224*224
        dataset = HelaDataset()
        shuffled_indices, num_train,ratio_samples_number = indices_shuffling(dataset)       
        testing_loader,training_loader,training_idx,testing_idx = split_train_test(dataset,shuffled_indices, num_train,ratio_samples_number,args.ratio)
            
      
    #print(args)
    train_organet(dataset,training_loader,testing_loader)

