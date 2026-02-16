import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
import torch.functional as F
import matplotlib.pyplot as plt
import random
import os
from PIL import Image
import torchvision.transforms as T
from torchvision import models
import torchvision.datasets as dset
import numpy as np
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


class SiameseCOCOWrapper(Dataset):
    def __init__(self,root,annFile,target_classes = None,transform = None):
        self.coco = dset.CocoDetection(root = root, annFile=annFile)
        self.transform = transform

        if target_classes:
            self.cat_ids = self.coco.coco.getCatIds(catNms = target_classes)
        else:
            self.cat_ids = self.coco.coco.getCatIds()


        self.label_to_data = {cat_id: [] for cat_id in self.cat_ids}


        valid_object = 0
        img_ids = self.coco.coco.getImgIds()

        for img_id in img_ids:
            ann_ids = self.coco.coco.getAnnIds(imgIds=img_id,catIds=self.cat_ids)
            anns = self.coco.coco.loadAnns(ann_ids)

            for ann in anns:

                x,y,w,h = ann["bbox"]
                if w < 15 or h < 15:
                    continue

                cat_id = ann["category_id"]

                if cat_id in self.label_to_data:
                    self.label_to_data[cat_id].append((img_id,ann["bbox"]))
                    valid_object += 1

        self.present_cat_ids = [k for k, v in self.label_to_data.items() if len(v) > 0]
        print(f"{valid_object} adet nesne toplandi")
    def __getitem__(self,index):

        cat_id1 = random.choice(self.present_cat_ids)
        img_id1,bbox1 = random.choice(self.label_to_data[cat_id1])


        should_get_same = (index % 2 == 0)

        if should_get_same:
            target = torch.tensor([1.0],dtype = torch.float32)

            img_id2, bbox2 = random.choice(self.label_to_data[cat_id1])

        else:
            target = torch.tensor([0.0],dtype = torch.float32)
            possible_negatives = [c for c in self.present_cat_ids if c!= cat_id1]

            if not possible_negatives:
                possible_negatives = [cat_id1]
            cat_id2 = random.choice(possible_negatives)
            img_id2, bbox2 = random.choice(self.label_to_data[cat_id2])

        img1 = self.load_crop_resize(img_id1,bbox1)
        img2 = self.load_crop_resize(img_id2,bbox2)

        return img1,img2, target

    def load_crop_resize(self, img_id, bbox):

        img_info = self.coco.coco.loadImgs(img_id)[0]
        path = os.path.join(self.coco.root,img_info["file_name"])


        image = Image.open(path).convert("RGB")

        x,y,w,h = bbox
        image = image.crop((x,y,x+w,y+h))

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return 3000

def get_data_loader(root_direct,ann_file,batch_size = 32):

    transform = T.Compose([
        T.Resize((224,224)), #ResNet 224x224 ister
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],std = [0.229,0.224,0.225])
    ])

    train_classes = ["person","car","dog","chair","cup","bottle"]
    test_classes = ["zebra","horse","banana"]


    train_dataset = SiameseCOCOWrapper(root_direct,ann_file,train_classes,transform)
    test_dataset = SiameseCOCOWrapper(root_direct,ann_file,test_classes,transform)

    train_dataloader = DataLoader(train_dataset,batch_size = 32,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size = 32,shuffle=False)

    return train_dataloader,test_dataloader

def visualize_data(loader, n=5):
    # Dataloader'dan bir batch çek
    img1, img2, target = next(iter(loader))

    # Figür oluştur
    fig, axes = plt.subplots(2, n, figsize=(15, 6))

    # --- ResNet Normalizasyonunu Geri Alma (Görüntü düzgün çıksın diye) ---
    # Bu değerler PyTorch standartlarıdır
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def unnormalize(tensor_img):
        # 1. (C, H, W) -> (H, W, C) yap (Matplotlib formatı)
        img = tensor_img.permute(1, 2, 0).numpy()
        # 2. Renkleri geri aç (x * std + mean)
        img = img * std + mean
        # 3. 0-1 arasına sıkıştır (Hata vermemesi için)
        return np.clip(img, 0, 1)

    for i in range(n):
        # --- Resim 1 (Üst Satır) ---
        # HATA 1 & 2 Düzeltildi: axes[0, i].imshow(...) kullanıldı ve boyut düzeltildi
        axes[0, i].imshow(unnormalize(img1[i]))
        axes[0, i].axis("off")

        # --- Resim 2 (Alt Satır) ---
        axes[1, i].imshow(unnormalize(img2[i]))
        axes[1, i].axis("off")

        # --- Başlık (HATA 3 Düzeltildi) ---
        # target[i].item() ile sadece o resmin etiketine bakıyoruz
        label = target[i].item()
        if label == 1.0:
            axes[0, i].set_title(f"Çift {i+1}: AYNI", color="green", fontweight="bold")
        else:
            axes[0, i].set_title(f"Çift {i+1}: FARKLI", color="red", fontweight="bold")

    plt.tight_layout()
    plt.show()

if "__main__" == __name__:
    root_dir = r"C:\Users\arpac\Desktop\2025-2026\DataSetss\train2017\train2017"
    ann_dir = r"C:\Users\arpac\Desktop\2025-2026\DataSetss\annotations_trainval2017\annotations\instances_train2017.json"
    train_dataloader,test_dataloader = get_data_loader(root_direct=root_dir,ann_file=ann_dir)
    visualize_data(train_dataloader,10)