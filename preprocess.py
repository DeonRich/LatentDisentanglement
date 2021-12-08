import torch
import numpy as np
import os
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image

preprocess = T.Compose([
   T.Resize(64),
   T.CenterCrop(64),
   T.ToTensor(),
])
def preprocessData():
	i=0
	folder_i = 0
	base = "data/Images"
	processed = "processed/"
	all_imgs = []
	all_labels = []
	for folder in os.listdir(base):
		for root, dirs, files in os.walk(os.path.join(base, folder)):
			for file in files:
				i += 1
				print(i)
				img = Image.open(os.path.join(root, file)).convert('RGB')
				x = preprocess(img)
				if x.max() > 1:
					raise Exception()
				all_imgs.append(x[None] * 2 - 1)
				all_labels.append(folder_i)
				os.makedirs(os.path.join(processed, folder), exist_ok=True)
				img = Image.fromarray((x.permute(1,2,0)*255).numpy().astype(np.uint8))
				img.save(os.path.join(processed, folder, file))
		folder_i += 1
	np.save("../PyTorch-GAN/implementations/infogan/pre_imgs", torch.cat(all_imgs).numpy())
	np.save("../PyTorch-GAN/implementations/infogan/pre_labels", torch.tensor(all_labels).numpy())
	print(torch.cat(all_imgs).numpy().shape)
	print(i)

def test():
	a = np.load("../PyTorch-GAN/implementations/infogan/pre_imgs.npy")
	b = np.load("../PyTorch-GAN/implementations/infogan/pre_labels.npy")
	dataset = TensorDataset(torch.tensor(a), torch.tensor(b).long())
	dl = DataLoader(dataset, batch_size=16)
	for img, label in dl:
		print(label)

if __name__ == '__main__':
	preprocessData()
	#test()