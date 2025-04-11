import zipfile

path_to_zip_file = "/home/y3229wan/scratch/project-9-at-2024-06-13-03-52-b472fec5.zip"
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall("Data")

