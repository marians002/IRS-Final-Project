import os


def map_names_to_ids(file_path=None):
    if file_path is None:
        # Get the directory of the current script
        script_dir = os.path.dirname(__file__)
        # Construct the absolute path to the sample_names.txt file
        file_path = os.path.join(script_dir, 'sample_names.txt')

    with open(file_path, 'r') as file:
        names = file.readlines()

    # Remove any leading/trailing whitespace characters (like newlines)
    names = [name.strip() for name in names]

    # Create a dictionary mapping names to IDs
    name_to_id = {name: i + 1 for i, name in enumerate(names)}

    return name_to_id
