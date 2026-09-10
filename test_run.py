import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models, torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import os, random, time
from collections import defaultdict

torch.manual_seed(42); random.seed(42)

# ── Config ─────────────────────────────────────────────
DATASET = r'c:\Users\riyat\OneDrive\Documents\palmvein\palm_vein_ready'
P, K    = 15, 4
EPOCHS  = 5
LR      = 1e-4
MARGIN  = 0.2
BATCHES = 20
EMB_DIM = 128
MEAN    = [0.485, 0.456, 0.406]
STD     = [0.229, 0.224, 0.225]

# ── Transforms ─────────────────────────────────────────
train_tf = T.Compose([
    T.Grayscale(3), T.RandomRotation(180, fill=0),
    T.RandomResizedCrop(224, scale=(0.85,1.15), ratio=(0.95,1.05)),
    T.ColorJitter(brightness=0.1), T.ToTensor(), T.Normalize(MEAN, STD)])
val_tf = T.Compose([T.Grayscale(3), T.ToTensor(), T.Normalize(MEAN, STD)])

# ── Dataset ────────────────────────────────────────────
class PalmDS(Dataset):
    def __init__(self, root, subjects, tf):
        self.tf = tf; self.samples = []; self.l2i = defaultdict(list)
        for lbl, s in enumerate(subjects):
            for f in sorted(os.listdir(os.path.join(root,s))):
                if f.endswith('.png'):
                    i = len(self.samples)
                    self.samples.append((os.path.join(root,s,f), lbl))
                    self.l2i[lbl].append(i)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, l = self.samples[i]
        return self.tf(Image.open(p)), l

class PKSampler(Sampler):
    def __init__(self, l2i, P, K, B):
        self.l2i=l2i; self.P=P; self.K=K; self.B=B; self.labels=list(l2i.keys())
    def __iter__(self):
        for _ in range(self.B):
            yield from [i for c in random.sample(self.labels,self.P)
                        for i in random.choices(self.l2i[c],k=self.K)]
    def __len__(self): return self.B*self.P*self.K

# ── Model ──────────────────────────────────────────────
class EmbNet(nn.Module):
    def __init__(self):
        super().__init__()
        bb = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for p in bb.parameters(): p.requires_grad=False
        for p in bb.layer4.parameters(): p.requires_grad=True
        self.bb   = nn.Sequential(*list(bb.children())[:-1])
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(512,EMB_DIM))
    def forward(self, x):
        return F.normalize(self.head(self.bb(x)), p=2, dim=1)
    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d)): m.eval()
        return self

def triplet_loss(emb, lbl):
    d = torch.cdist(emb,emb).pow(2)
    pm = (lbl.unsqueeze(1)==lbl.unsqueeze(0)); pm.fill_diagonal_(False)
    nm = ~(lbl.unsqueeze(1)==lbl.unsqueeze(0))
    hp = (d*pm.float()).max(1).values
    dn = d.clone(); dn[~nm]=1e9
    hn = dn.min(1).values
    return F.relu(hp-hn+MARGIN).mean()

@torch.no_grad()
def eval_nn(model, ds):
    model.eval()
    loader = DataLoader(ds, batch_size=32)
    embs, lbls = [], []
    for x,l in loader:
        embs.append(model(x)); lbls.extend(l.tolist())
    emb = torch.cat(embs); lbl = torch.tensor(lbls)
    d = torch.cdist(emb,emb); d.fill_diagonal_(1e9)
    return (lbl[d.argmin(1)]==lbl).float().mean().item()

# ── Run ────────────────────────────────────────────────
subjects = sorted([d for d in os.listdir(DATASET) if os.path.isdir(os.path.join(DATASET,d))])
tr_ds = PalmDS(DATASET, subjects[:48], train_tf)
vl_ds = PalmDS(DATASET, subjects[48:], val_tf)
loader = DataLoader(tr_ds, batch_size=P*K, sampler=PKSampler(tr_ds.l2i,P,K,BATCHES))

model = EmbNet()
opt   = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=LR)

print(f'Subjects: {len(subjects)} | Train: {len(tr_ds)} imgs | Val: {len(vl_ds)} imgs')
print(f'Running {EPOCHS} epochs to verify pipeline...')
print('-'*50)

for ep in range(1, EPOCHS+1):
    t0 = time.time()
    model.train(); tot=0; nb=0
    for imgs, lbls in loader:
        opt.zero_grad()
        loss = triplet_loss(model(imgs), lbls)
        loss.backward(); opt.step()
        tot+=loss.item(); nb+=1
    acc = eval_nn(model, vl_ds)
    print(f'Epoch {ep}/{EPOCHS}  loss={tot/nb:.4f}  val_acc={acc*100:.1f}%  ({time.time()-t0:.0f}s)')

print('-'*50)
print('Done. If loss decreased → pipeline is working correctly.')
