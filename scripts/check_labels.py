import os

"""
Recursively scans a root directory for all subdirectories named labels.
Checks if each labels directory is empty or non-empty.
Prints summaries and lists of empty and non-empty label folders.
"""

def check_labels_empty(root_directory):
    """
    Recursively search for subdirectories named 'labels' within root_directory
    and return two lists: one for empty labels folders and one for non-empty folders.
    """
    empty_labels = []
    non_empty_labels = []
    
    for dirpath, dirnames, filenames in os.walk(root_directory):
        # Check if the current directory is a 'labels' folder (case insensitive)
        if os.path.basename(dirpath).lower() == "labels":
            # List all items inside the labels folder
            items = os.listdir(dirpath)
            if not items:
                empty_labels.append(dirpath)
            else:
                non_empty_labels.append(dirpath)
    
    return empty_labels, non_empty_labels

def main():
    output_base = "/home/thibault/ProcessedDynamicEarthNet"
    
    empty_labels, non_empty_labels = check_labels_empty(output_base)
    
    empty_count = len(empty_labels)
    non_empty_count = len(non_empty_labels)
    
    print("=== Labels Directories Summary ===")
    print(f"Total 'labels' directories found: {empty_count + non_empty_count}")
    print(f"Empty 'labels' directories: {empty_count}")
    print(f"Non-empty 'labels' directories: {non_empty_count}\n")
    
    if empty_labels:
        print("=== List of Empty 'labels' Directories ===")
        for folder in empty_labels:
            print(folder)
    
    if non_empty_labels:
        print("\n=== List of Non-Empty 'labels' Directories ===")
        for folder in non_empty_labels:
            print(folder)

if __name__ == "__main__":
    main()
