import torch
import os

def inspect_tensor(tensor):
    """
    Prints various properties of a PyTorch tensor.
    """
    print("Tensor content:")
    print(tensor)
    print("\nProperties:")
    print(f"Shape: {tensor.shape}")
    print(f"Data type: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Requires grad: {tensor.requires_grad}")
    print(f"Number of elements: {tensor.numel()}")
    print(f"Element size (bytes): {tensor.element_size()}")
    print(f"Total memory size (bytes): {tensor.element_size() * tensor.numel()}")
    print(f"Number of dimensions: {tensor.dim()}")
    print(f"Strides: {tensor.stride()}")

if __name__ == "__main__":
    tensor_path = '/media/thibault/DynEarthNet/datasets/unet/planet.10N/1311_3077_13/2018-01/pixel_dataset_2018-01.pt'

    if os.path.exists(tensor_path):
        loaded_obj = torch.load(tensor_path)
        # If it is a tensor, inspect it
        if isinstance(loaded_obj, torch.Tensor):
            print("Loaded object is a tensor.")
            inspect_tensor(loaded_obj)
        # If it is a dictionary, try to find and inspect tensor(s) within it
        elif isinstance(loaded_obj, dict):
            print("Loaded object is a dictionary. Attempting to inspect tensor values...")
            tensor_found = False
            for key, value in loaded_obj.items():
                if isinstance(value, torch.Tensor):
                    tensor_found = True
                    print(f"\nInspecting tensor at key: '{key}'")
                    inspect_tensor(value)
            if not tensor_found:
                print("No tensor found in the dictionary.")
        else:
            print("The loaded object is not a PyTorch tensor or a dictionary containing tensors.")
    else:
        print("File not found. Please check the path and try again.")
