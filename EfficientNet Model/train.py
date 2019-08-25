import torch
import torch.nn as nn
from apex import amp
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import time

from ..trainer.train import train_model , test_model
from ..model.model import EfficientNet
from ..data_loader.data_loaders import APTOSDATA



num_classes = 1
lr = 1e-3


train      = '../input/aptos2019-blindness-detection/train_images/'
test       = '../input/aptos2019-blindness-detection/test_images/'
train_csv  = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
train_df, val_df = train_test_split(train_csv, test_size=0.1, random_state=2018, stratify=train_csv.diagnosis)
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)



model = EfficientNet.from_name('efficientnet-b0')
model.load_state_dict(torch.load('../input/efficientnet-pytorch/efficientnet-b0-08094119.pth'))
in_features = model._fc.in_features
model._fc = nn.Linear(in_features, num_classes)
model.cuda()



optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-120, 120)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

trainset     = APTOSDATA(train_df, transform =train_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
valset       = APTOSDATA(val_df, transform   =train_transform)
val_loader   = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False, num_workers=4)




best_avg_loss = 100.0
n_epochs      = 10


for epoch in range(n_epochs):
    
    print('lr:', scheduler.get_lr()[0]) 
    start_time   = time.time()
    avg_loss     = train_model(epoch , model, train_loader , optimizer , criterion )
    avg_val_loss = test_model(model  , val_loader , criterion)
    elapsed_time = time.time() - start_time 
    print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
        epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))
    
    if avg_val_loss < best_avg_loss:
        best_avg_loss = avg_val_loss
        torch.save(model.state_dict(), 'weight_best.pt')
    
    scheduler.step()