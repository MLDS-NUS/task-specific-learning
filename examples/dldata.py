# %%
import os, requests
from taskspec.utils import find_project_root

# %%
headers = {}
url = "https://huggingface.co/datasets/MLDS-NUS/task-specific-learning/resolve/main/"

# %%
def download_file(filename, load_path, save_path):
    response = requests.get(url + load_path + filename, headers=headers)
    response.raise_for_status()
    os.makedirs(save_path, exist_ok=True) 
    if os.path.exists(save_path + filename):
        print(f"{filename} already exists in {save_path}.")
        return
    else:
        print(f"Download {filename} to {save_path}.")
        with open(save_path + filename, "wb") as f:
            f.write(response.content)

# %%
load_path = "Eg1data/data/"
save_path = find_project_root() + '/examples/Eg1/data/data/'
filename_list = [f"E1_dr{randomseed}.npz" for randomseed in range(20)]
for filename in filename_list:
    download_file(filename, load_path, save_path)

# %%
load_path = "Eg1data/data_as/"
save_path = find_project_root() + '/examples/Eg1/data/data_as/'
filename_list = [f"E1_as_dr{alpha}_{randomseed}.npz" for randomseed in range(20) for alpha in [0.0, 0.25, 0.5, 0.75, 0.99]]
for filename in filename_list:
    download_file(filename, load_path, save_path)

# %%
load_path = "Eg2data/data/"
save_path = find_project_root() + '/examples/Eg2/data/data/'
filename_list = [f"E2_dr{randomseed}.npz" for randomseed in range(20)]
for filename in filename_list:
    download_file(filename, load_path, save_path)

# %%
load_path = "Eg3data/data/"
save_path = find_project_root() + '/examples/Eg3/data/data/'
filename_list = [f"E3_d{mep}_{D}_{randomseed}.npz" for randomseed in range(20) for mep in [0, 25, 50, 75] for D in [0.1, 0.2, 1.0]]
for filename in filename_list:
    download_file(filename, load_path, save_path)


