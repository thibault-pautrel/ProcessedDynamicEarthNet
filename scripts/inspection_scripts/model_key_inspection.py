import torch
from basic_spdnet_pipeline import SPDNet2BiRe

# Create an example model
model = SPDNet2BiRe(input_dim=112, num_classes=4, use_batch_norm=True)

# Print all parameter keys
print("\nModel state_dict keys:")
for k in model.state_dict().keys():
    print(k)
