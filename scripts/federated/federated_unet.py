import os
import sys
import torch
sys.path.append("/home/thibault/ProcessedDynamicEarthNet/scripts")  
from unet_pipeline import UNetLite, train_model, get_dataloader, UNetCropDataset, set_seed

def average_model_weights(model_paths):
    avg_state_dict = None
    n_models = len(model_paths)

    for path in model_paths:
        state_dict = torch.load(path, map_location='cpu')
        if avg_state_dict is None:
            avg_state_dict = {
                k: v.clone().float() for k, v in state_dict.items()
            }
        else:
            for k in avg_state_dict:
                avg_state_dict[k] += state_dict[k].float()

    for k in avg_state_dict:
        avg_state_dict[k] /= n_models

    return avg_state_dict


def train_one_client(client_name, data_dir, global_weights, output_dir, round_idx):
    print(f"\n--- Training client: {client_name} ---")
    dataset = UNetCropDataset(split_dir=data_dir, max_T=28, augment=True)
    size = len(dataset)
    train_len = int(size * 0.7)
    val_len = int(size * 0.15)
    test_len = size - train_len - val_len
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = get_dataloader(train_ds, batch_size=1, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=1, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=1, shuffle=False)

    model = UNetLite(in_channels=112, num_classes=4, dropout_p=0.1)
    if global_weights:
        model.load_state_dict(global_weights)

    model_name = f"{client_name}_round{round_idx}"
    train_model(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        epochs=5,  # short for demo
        lr=1e-4,
        num_classes=4,
        weight_decay=5e-4
    )

    ckpt_path = os.path.join("/home/thibault/ProcessedDynamicEarthNet/checkpoints", f"{model_name}_best.pt")
    return ckpt_path

if __name__ == "__main__":
    set_seed(42)
    
    client_dirs = {
        #"urban": "/media/thibault/DynEarthNet/federated/datasets/urban",
        "mixed": "/media/thibault/DynEarthNet/federated/datasets/mixed",
        "forest": "/media/thibault/DynEarthNet/federated/datasets/forest",
        "water": "/media/thibault/DynEarthNet/federated/datasets/water"
    }

    NUM_ROUNDS = 3
    global_weights = None

    for round_idx in range(NUM_ROUNDS):
        print(f"\n================ Federated Round {round_idx} ================\n")
        client_ckpts = []
        for client_name, data_dir in client_dirs.items():
            ckpt = train_one_client(client_name, data_dir, global_weights, "./checkpoints", round_idx)
            client_ckpts.append(ckpt)

        global_weights = average_model_weights(client_ckpts)
        print(f"\n[ROUND {round_idx}] Aggregated global model from {len(client_ckpts)} clients.\n")
