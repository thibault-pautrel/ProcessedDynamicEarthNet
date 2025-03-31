"""
test.py

This script demonstrates how to load a PyTorch object (either a tensor or a dictionary containing tensors)
from a specified file path and prints detailed information about the tensor(s).

The script:
  - Loads a file from disk using torch.load.
  - Checks if the loaded object is a tensor, and if so, prints its content and various properties.
  - If the loaded object is a dictionary, it attempts to locate any tensor values within it and prints their properties.
  - Provides useful diagnostic information such as shape, data type, device, gradient requirement, and memory usage.

Usage:
    Simply run the script. Ensure that the tensor file path (tensor_path) is correctly set to a valid .pt file.
"""
import torch
import os

def inspect_tensor(tensor):
    """
    Print detailed information about a PyTorch tensor.

    This function prints:
      - The tensor's contents.
      - Shape, data type, device (CPU/GPU), and gradient requirement.
      - Number of elements and memory usage in bytes.
      - Number of dimensions and the strides of the tensor.

    Args:
        tensor (torch.Tensor): The tensor to be inspected.
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
    tensor_path = '/media/thibault/DynEarthNet/subsampled_data/datasets/unet/test/1286_2921_13/2018-01/pixel_dataset_2018-01.pt'

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
