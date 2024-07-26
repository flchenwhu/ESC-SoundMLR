import torch.nn as nn


class ProjectionModel(nn.Module):
	def __init__(self):
		super().__init__()
		# self.fc = nn.Linear(2048, 64)
		self.fc = nn.Linear(1280, 64)
		# self.fc = nn.Linear(33280, 64)


	def forward(self, x):
		x = self.fc(x)

		return x

    
    
