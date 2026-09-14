import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from oddball import Dataset, load


class MaskTransform(nn.Module):
    def __init__(self,dim,hidden=64,mode="mul"):
         super().__init__()
         self.mode=mode
         self.net=nn.Sequential(
                  nn.Linear(dim,hidden,bias=False),
                  nn.ReLU(),
                  nn.Linear(hidden,dim,bias=False)
         )

    def forward(self,x):
         m=self.net(x)
         if self.mode=="mul":
                  m=torch.sigmoid(m)
                  return m*x
         else:
                  return m+x


class Encoder(nn.Module):
    def __init__(self,dim,hidden=32,out_dim=24,depth=2):
         super().__init__()
         layers=[]
         d=dim
         for i in range(depth):
                  layers.append(nn.Linear(d,hidden,bias=False))
                  layers.append(nn.ReLU())
                  d=hidden
         layers.append(nn.Linear(d,out_dim,bias=False))
         self.net=nn.Sequential(*layers)

    def forward(self,x):
         return self.net(x)


class NeuTraLAD(nn.Module):
    def __init__(self,dim,n_trans=11,hidden=64,out_dim=32,mode="mul",temperature=0.1):
         super().__init__()
         self.n_trans=n_trans
         self.temperature=temperature
         self.transforms=nn.ModuleList([MaskTransform(dim,hidden,mode) for _ in range(n_trans)])
         self.encoder=Encoder(dim,hidden,out_dim)

    def forward(self,x):
         z0=self.encoder(x)
         zs=[self.encoder(t(x)) for t in self.transforms]
         return z0,zs

    def dcl_loss(self,z0,zs):
         eps=1e-8
         z0n=F.normalize(z0,dim=1)
         zsn=[F.normalize(z,dim=1) for z in zs]
         K=len(zsn)
         sims_ref=[torch.exp(torch.sum(zsn[k]*z0n,dim=1)/self.temperature) for k in range(K)]
         losses=torch.zeros(z0.shape[0],device=z0.device)
         for k in range(K):
                  denom=torch.zeros(z0.shape[0],device=z0.device)
                  for l in range(K):
                           if l==k:
                                    continue
                           sim_kl=torch.exp(torch.sum(zsn[k]*zsn[l],dim=1)/self.temperature)
                           denom=denom+sim_kl
                  term=-torch.log(sims_ref[k]/(denom+eps)+eps)
                  losses=losses+term
         return losses

    def score(self,x):
         z0,zs=self.forward(x)
         return self.dcl_loss(z0,zs)


def train_neutral_ad(model,train_loader,epochs=20,lr=1e-3,device="cpu",x_val=None,y_val=None,patience=15):
         opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-3)
         model.to(device)
         best_auc=-1.0
         best_state=None
         stale=0
         for ep in range(epochs):
                  model.train()
                  total=0.0
                  count=0
                  for batch in train_loader:
                           x=batch[0].to(device)
                           opt.zero_grad()
                           z0,zs=model(x)
                           loss=model.dcl_loss(z0,zs).mean()
                           loss.backward()
                           opt.step()
                           total=total+loss.item()*x.shape[0]
                           count=count+x.shape[0]
                  msg=f"epoch {ep+1}/{epochs} loss {total/count:.4f}"
                  if x_val is not None:
                           val_auc,_=evaluate_auc(model,x_val,y_val,device=device)
                           msg=msg+f" val_auc {val_auc:.4f}"
                           if val_auc>best_auc:
                                    best_auc=val_auc
                                    best_state={k:v.clone() for k,v in model.state_dict().items()}
                                    stale=0
                           else:
                                    stale=stale+1
                  print(msg)
                  if x_val is not None and stale>=patience:
                           print(f"early stopping at epoch {ep+1}, best val_auc {best_auc:.4f}")
                           break
         if best_state is not None:
                  model.load_state_dict(best_state)
         return model


def evaluate_auc(model,x_test,y_test,device="cpu"):
         model.eval()
         with torch.no_grad():
                  x=torch.tensor(x_test,dtype=torch.float32).to(device)
                  scores=model.score(x).cpu().numpy()
         auc=roc_auc_score(y_test,scores)
         return auc,scores


def load_odds_style_csv(path,label_col="label"):
         import pandas as pd
         df=pd.read_csv(path)
         y=df[label_col].values
         x=df.drop(columns=[label_col]).values
         return x,y


def run_experiment(x_normal_train,x_test,y_test,dim,n_trans=11,epochs=20,mode="mul",batch_size=128,device="cpu",use_val=True,patience=15,val_frac=0.1):
         scaler=StandardScaler()
         x_normal_train=scaler.fit_transform(x_normal_train)
         x_test=scaler.transform(x_test)
         x_val=None
         y_val=None
         if use_val:
                  n=x_normal_train.shape[0]
                  idx=np.random.permutation(n)
                  n_val=int(val_frac*n)
                  val_idx=idx[:n_val]
                  train_idx=idx[n_val:]
                  x_val_normal=x_normal_train[val_idx]
                  x_normal_train=x_normal_train[train_idx]
                  n_anom_val=max(1,int(val_frac*x_test.shape[0]))
                  anom_pool=np.where(y_test==1)[0]
                  anom_val_idx=np.random.choice(anom_pool,size=min(n_anom_val,anom_pool.shape[0]),replace=False)
                  x_val_anom=x_test[anom_val_idx]
                  x_val=np.concatenate([x_val_normal,x_val_anom],axis=0)
                  y_val=np.concatenate([np.zeros(x_val_normal.shape[0]),np.ones(x_val_anom.shape[0])])
         train_ds=TensorDataset(torch.tensor(x_normal_train,dtype=torch.float32))
         train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,drop_last=True)
         model=NeuTraLAD(dim=dim,n_trans=n_trans,mode=mode)
         model=train_neutral_ad(model,train_loader,epochs=epochs,device=device,x_val=x_val,y_val=y_val,patience=patience)
         auc,scores=evaluate_auc(model,x_test,y_test,device=device)
         print(f"test AUC {auc:.4f}")
         return model,auc,scores


def load_thyroid_via_oddball():
         x,y=load(Dataset.THYROID)
         y=np.asarray(y).reshape(-1)
         x=np.asarray(x)
         normal_mask=(y==0)
         anom_mask=(y==1)
         x_normal=x[normal_mask]
         x_anom=x[anom_mask]
         n_normal=x_normal.shape[0]
         idx=np.random.permutation(n_normal)
         x_normal=x_normal[idx]
         split=int(0.7*n_normal)
         x_train=x_normal[:split]
         x_test=np.concatenate([x_normal[split:],x_anom],axis=0)
         y_test=np.concatenate([np.zeros(n_normal-split),np.ones(x_anom.shape[0])])
         return x_train,x_test,y_test


if __name__=="__main__":
         np.random.seed(0)
         torch.manual_seed(0)
         device="cuda" if torch.cuda.is_available() else "cpu"
         x_train,x_test,y_test=load_thyroid_via_oddball()
         dim=x_train.shape[1]
         print("train shape",x_train.shape,"test shape",x_test.shape,"n anomalies",int(y_test.sum()))
         model,auc,scores=run_experiment(x_train,x_test,y_test,dim=dim,n_trans=11,epochs=200,mode="res",batch_size=64,device=device)
         print("final auc",auc)