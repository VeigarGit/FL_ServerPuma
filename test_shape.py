import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 8)

model = MyModel()
new_state = MyModel()
new_state.fc.weight = nn.Parameter(torch.randn(5, 10)) # Rank went from 8 to 5

for new_param, old_param in zip(new_state.parameters(), model.parameters()):
    old_param.data = new_param.data.clone()

print(model.fc.weight.shape)
