import os
import zipfile
import urllib.request
import yaml

def prepare_dataset():
    dataset_dir = "dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    zip_path = os.path.join(dataset_dir, "malpractice_data.zip")

    # Download exam malpractice dataset
    url = "https://universe.roboflow.com/ds/7z8g2fU0Zq?key=6T3c6i3gXq"
    downloaded = False
    if not os.path.exists(zip_path):
        try:
            print(f"Attempting to download dataset from {url}...")
            # Set a request with User-Agent
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response, open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            print("Successfully extracted downloaded dataset.")
            downloaded = True
        except Exception as e:
            print(f"Fallback to bundled sample dataset: {e}")
    else:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            downloaded = True
        except Exception:
            pass

    if not downloaded:
        # Fallback to local bundled dataset.zip
        local_bundle = "dataset.zip"
        if os.path.exists(local_bundle):
            print(f"Extracting local verified dataset from {local_bundle} into {dataset_dir}...")
            with zipfile.ZipFile(local_bundle, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
        elif os.path.exists("exam_dataset"):
            print("Copying verified dataset from exam_dataset/ into dataset/...")
            import shutil
            for item in os.listdir("exam_dataset"):
                s = os.path.join("exam_dataset", item)
                d = os.path.join(dataset_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)

    # Ensure dataset/data.yaml exists and has absolute path
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    abs_dataset_dir = os.path.abspath(dataset_dir).replace("\\", "/")
    
    yaml_config = {
        "path": abs_dataset_dir,
        "train": "train/images",
        "val": "val/images",
        "nc": 3,
        "names": ["exam_chit", "cell_phone", "book"]
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(yaml_config, f, sort_keys=False)

    print(f"Dataset preparation complete! Data configuration saved to {data_yaml_path}")
    return data_yaml_path

if __name__ == "__main__":
    prepare_dataset()
