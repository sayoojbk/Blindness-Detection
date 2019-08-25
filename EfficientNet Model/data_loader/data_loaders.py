import cv2
import numpy as np 
from torch.utils.data import Dataset
from ..utils.util import expand_path , crop_image_from_gray , crop_image1


IMG_SIZE    = 256

class APTOSDATA(Dataset):
    
    def __init__(self, dataframe, transform=None ):
        self.df = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        label = self.df.diagnosis.values[idx]
        label = np.expand_dims(label, -1)
        
        p = self.df.id_code.values[idx]
        p_path = expand_path(p)
        image = cv2.imread(p_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 30) ,-4 ,128)
        
        # image = transforms.ToPILImage()(image)
        
        if self.transform:
            image = self.transform(image)

        return image, label


