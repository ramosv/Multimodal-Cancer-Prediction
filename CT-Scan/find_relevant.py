from pathlib import Path
import pandas as pd
import shutil

def main():
    root = Path.cwd()
    try:
        clinical = pd.read_csv(root / "clinical_filtered.csv", index_col=0, encoding="utf-8")
    except UnicodeDecodeError:
            print("Retrying with ISO-8859-1 encoding for proteomics data...")
            clinical = pd.read_csv(root / "clinical_filtered.csv", index_col=0, encoding="ISO-8859-1")
    patient_ids = set(clinical.index)
    cptac_dir = root / "cptac-ccrcc"

    present_dir = root / "present"
    not_present_dir = root / "not_present"
    present_dir.mkdir(exist_ok=True)
    not_present_dir.mkdir(exist_ok=True)

    for folder in cptac_dir.iterdir():
        if folder.is_dir():
            if folder.name in patient_ids:
                shutil.move(str(folder), str(present_dir / folder.name))
            else:
                shutil.move(str(folder), str(not_present_dir / folder.name))

    num_present = len(list(present_dir.iterdir()))
    num_not_present = len(list(not_present_dir.iterdir()))
    print(f"Moved {num_present} folders to {present_dir}")
    print(f"Moved {num_not_present} folders to {not_present_dir}")

if __name__ == "__main__":
    main()
