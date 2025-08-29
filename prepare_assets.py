import os
import requests
from tqdm import tqdm

def download_file(url: str, fpath: str):
    if os.path.exists(fpath):
        print(f"File already exists at {fpath}. Skipping download.")
        return
        
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(fpath, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True,
                desc=f"Downloading {os.path.basename(fpath)}"
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"Successfully downloaded {os.path.basename(fpath)}.")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to download {url}. Reason: {e}")
        if os.path.exists(fpath):
            os.remove(fpath)
        raise

def prepare_data(data_dir: str):
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    
    if not os.path.exists(train_path):
        train_url = "https://huggingface.co/datasets/TokenBender/avataRL/resolve/main/train.bin"
        download_file(train_url, train_path)
        
    if not os.path.exists(val_path):
        val_url = "https://huggingface.co/datasets/TokenBender/avataRL/resolve/main/val.bin"
        download_file(val_url, val_path)

def prepare_critic_model(critic_path: str):
    if not os.path.exists(critic_path):
        filename = os.path.basename(critic_path)
        critic_url = f"https://huggingface.co/TokenBender/avataRL-critic/resolve/main/{filename}"
        download_file(critic_url, critic_path)
        
    print("Critic model is ready.")