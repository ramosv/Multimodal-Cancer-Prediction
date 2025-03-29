from pathlib import Path
import pandas as pd
import shutil

def main():
    root = Path.cwd()
    print(f"Current working dir: {root}")
    clinical = pd.read_csv(root/"jessica_output/clinical_filtered.csv", index_col=0)
    patient_ids = set(clinical.index)

    print(f"Unique Patients clinical: {len(patient_ids)}")
    #breakpoint()
    cptac_dir = root / "CT-Scan/CPTAC-CCRCC"

    present_dir = root / "CT-Scan/present"
    not_present_dir = root / "CT-Scan/not_present"
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
