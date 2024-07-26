import torch.nn as nn

import config


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc = nn.Linear(2048, config.class_numbers)
        self.fc = nn.Linear(1280, config.class_numbers)






    def forward(self, x):
        x = self.fc(x)

        return x


