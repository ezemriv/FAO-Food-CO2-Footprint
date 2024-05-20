import os

def rename_with_prefix(folder_path):
    """Renames all files in a folder with a sequential prefix and the original base name."""

    file_list = os.listdir(folder_path)
    file_count = len(file_list)

    for index, filename in enumerate(file_list, start=1):
        prefix = f"{index:02d}"
        base_name, extension = os.path.splitext(filename)  # Split base name and extension
        new_filename = f"{prefix}-{base_name}{extension}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)

if __name__ == "__main__":
    folder_path = input("Enter the folder path: ")
    rename_with_prefix(folder_path)
    print("Files renamed successfully!")
