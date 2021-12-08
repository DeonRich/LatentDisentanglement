import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as ptd
import numpy as np
from model import *
from PIL import Image
import time

def getEntropy(probs):
	masked_logprob = torch.where(probs == 0, torch.zeros_like(probs), probs.log())
	return -(probs * masked_logprob).sum()

class Encoder(nn.Module):
	def __init__(self, out_dim, ndf=1024):
		super().__init__()

		self.block1 = DBlockOptimized(3, ndf >> 4)
		self.block2 = DBlock(ndf >> 4, ndf >> 3, downsample=True)
		self.block3 = DBlock(ndf >> 3, ndf >> 2, downsample=True)
		self.block4 = DBlock(ndf >> 2, ndf >> 1, downsample=True)
		self.block5 = DBlock(ndf >> 1, ndf, downsample=True)
		self.l6 = SNLinear(ndf, out_dim)
		self.activation = nn.ReLU(True)

		nn.init.xavier_uniform_(self.l6.weight.data, 1.0)

	def forward(self, x):
		h = x
		h = self.block1(h)
		h = self.block2(h)
		h = self.block3(h)
		h = self.block4(h)
		h = self.block5(h)
		h = self.activation(h)
		h = torch.sum(h, dim=(2, 3))
		y = self.l6(h)
		return y

class mlp(nn.Module):
	def __init__(self, input_dim, hidden_dims, out_dim):
		super(mlp, self).__init__()
		input_dims = [input_dim] + list(hidden_dims)
		self.layers = nn.ModuleList([nn.Linear(input_dims[i], dim) for i, dim in enumerate(hidden_dims)])
		self.out_layer = nn.Linear(input_dims[-1], out_dim)

	def forward(self, x):
		for layer in self.layers:
			x = F.tanh(layer(x))
		return self.out_layer(x)

class Disco(object):
	def __init__(self):
		self.eps = 1e-2
		self.nz  = 128
		self.B = self.N = self.M  = 8
		self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		self.navigator = mlp(self.nz, [], self.nz).to(self.device)
		self.encoder   = Encoder(self.nz).to(self.device)
		self.generator = Generator64().to(self.device)
		state_dict = torch.load("out/baseline-64-295k/ckpt/295000.pth")
		self.generator.load_state_dict(state_dict["net_g"])
		self.temp = 1
		self.iterations = int(7000)
		self.opt = torch.optim.Adam(par for model in [self.navigator, self.encoder] for par in model.parameters())
	
	def getVariationDistance(self, z, eps, dirs):
		v = self.encoder(self.generator(z + eps * self.navigator(dirs))) - self.encoder(self.generator(z))
		return F.normalize(v).abs()
	
	def sampleLatentSpace(self, batch):
		return torch.randn(batch, self.nz, device=self.device)
	
	def getEps(self, size):
		return 2*self.eps*torch.rand(size, 1, device=self.device)-self.eps

	def getDirections(self):
		pos = torch.multinomial(torch.ones(self.nz, device=self.device), 1, replacement=True)
		neg = torch.multinomial(torch.ones(self.nz-1, device=self.device), self.M, replacement=True)
		neg = torch.where(pos == neg, self.nz - 1, neg)
		pos = F.one_hot(pos, self.nz)
		neg = F.one_hot(neg, self.nz)
		return pos.float(), neg.float()

	def get_loss(self):
		z, z_pos, z_neg = self.sampleLatentSpace(self.B), self.sampleLatentSpace(self.N), self.sampleLatentSpace(self.M)
		e, e_pos, e_neg = self.getEps(self.B), self.getEps(self.N), self.getEps(self.M)
		p_dir, n_dir = self.getDirections()
		q	 = self.getVariationDistance(z, e, p_dir)
		k_pos = self.getVariationDistance(z_pos, e_pos, p_dir)
		k_neg = self.getVariationDistance(z_neg, e_neg, n_dir)
		l_pos = F.logsigmoid((q @ k_pos.T) / self.temp).sum(1)
		l_neg = F.logsigmoid((q @ k_neg.T) / self.temp).sum(1)
		l_logits = (l_pos - l_neg).mean()
		c_probs = F.normalize(torch.cat([q, k_pos]).mean(0, keepdim=True), p=1)
		l_ed  = -getEntropy(c_probs)
		loss = -(l_logits + l_ed)
		return loss

	def train_step(self):
		self.opt.zero_grad()
		loss = self.get_loss()
		loss.backward()
		self.opt.step()
		return loss.item()

	def train(self):
		history = []
		for i in range(self.iterations):
			loss = self.train_step()
			if i % 10 == 0:
				print(f"iter: {i}, {loss}")
				history.append(loss)
		torch.save(self.navigator.state_dict(), "nav.weights")
		torch.save(self.encoder.state_dict(), "enc.weights")
		return history

def main():
	start = time.time()
	disco  = Disco()
	history = disco.train()
	end = time.time()
	print(f"{disco.iterations} iters in {(end-start) / 60:0.3} minutes")
	np.save("history", np.array(history))
	#plt.plot(history)
	#plt.savefig("fig.png")
def testDirections():
	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	sample = torch.eye(128, device=device)
	navigator = mlp(128, [], 128).to(device)
	navigator.load_state_dict(torch.load("saved/nav.weights"))
	directions = F.normalize(navigator(sample))
	dots = directions @ directions.T
	dots = dots.abs().mean(1).detach().cpu().numpy()
	print(np.mean(dots), np.std(dots))
def create_pics():
	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	#z = torch.tile(torch.randn(1,128), (8,1)).to(device)
	z = torch.zeros(8,128).to(device)
	#z_dir = torch.zeros(1,6,128).to(device)
	navigator = mlp(128, [], 128).to(device)
	navigator.load_state_dict(torch.load("saved/nav.weights"))
	G = Generator64().to(device)
	state_dict = torch.load("out/baseline-64-295k/ckpt/295000.pth")
	eps = 1
	eps = eps*torch.arange(6, device=device).reshape((1,6,1))/5
	direction = torch.eye(128, device=device)
	direction = direction[48:56]
	base, c = [], []
	for s in range(6):
	#z_dir = torch.zeros(8, 6, 128).to(device)
		z_dir = eps * F.normalize(navigator(direction.float()))[:, None]
		z_prime = z[:, None] + z_dir
		print(z_prime.shape)
		z_prime = z_prime.reshape((-1, 128))
		G.load_state_dict(state_dict["net_g"])
		im1 = ((G(z_prime[np.arange(8)*6+s]) + 1)*127.5).permute((0,2,3,1)).detach().cpu().numpy()
		im_h1 = np.concatenate([im1[i] for i in range(8)]).astype(np.uint8)
		c.append(im_h1)
		#img = Image.fromarray(im_h1, "RGB")
		#img.save(f"pics2/c_{s}.png")

		z_dir = eps * direction.float()[:, None]
		z_prime = z[:, None] + z_dir
		z_prime = z_prime.reshape((-1, 128))
		
		G.load_state_dict(state_dict["net_g"])
		im = ((G(z_prime[np.arange(8)*6+s]) + 1)*127.5).permute((0,2,3,1)).detach().cpu().numpy()
		im_h = np.concatenate([im[i] for i in range(8)]).astype(np.uint8)
		base.append(im_h)
		#img = Image.fromarray(im_h, 'RGB')
		#img.save(f"pics2/base_{s}.png")
	img = Image.fromarray(np.concatenate(base, 1), "RGB")
	img.save(f"pics2/base.png")
	img = Image.fromarray(np.concatenate(c, 1), "RGB")
	img.save(f"pics2/c.png")



if __name__ == '__main__':
	#main()
	create_pics()
	#testDirections()

