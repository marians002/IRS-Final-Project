def map_names_to_ids(file_path='sample_names.txt'):
    with open(file_path, 'r') as file:
        names = file.readlines()

    # Remove any leading/trailing whitespace characters (like newlines)
    names = [name.strip() for name in names]

    # Create a dictionary mapping names to IDs (Angelica Powers: 1)
    name_to_id = {name: i+1 for i, name in enumerate(names)}

    return name_to_id
