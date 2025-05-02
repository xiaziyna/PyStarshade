import os
from importlib import resources

package_path = resources.files('pystarshade')
DATA_DIR = os.path.abspath(os.path.join(package_path, 'data'))
# Scenes directory
SCENES_DIR = os.path.join(DATA_DIR, 'scenes')

HOME_DIR = os.path.expanduser("~")
OUTPUT_DIR = os.path.join(HOME_DIR, "PyStarshade")
os.makedirs(OUTPUT_DIR, exist_ok=True)
