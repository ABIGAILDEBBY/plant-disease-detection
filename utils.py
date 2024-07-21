import os


def total_files(folder_path):
    """
    Counts the number of files in a given folder path.

    Args:
        folder_path (str): Path to the directory containing files.

    Returns:
        int: The number of files in the folder.
    """
    num_files = len(
        [
            f
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
        ]
    )
    return num_files
