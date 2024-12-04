import os
from urllib.request import urlretrieve

def exovista_scenes_file():
    filename = "Scene.py"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, '..', 'pystarshade', 'data', 'scenes'))
    file_path = os.path.join(data_dir, filename)
    download_url = "https://raw.githubusercontent.com/alexrhowe/ExoVista/main/Scene.py"

    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(file_path):
        try:
            urlretrieve(download_url, file_path)
            print(f"{filename} downloaded to {data_dir}.")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
    else:
        print(f"{filename} already exists in {data_dir}.")
