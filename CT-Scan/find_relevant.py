from pathlib import Path
import pandas as pd
import shutil

def main():
    root = Path.cwd()
    # print(f"Current working dir: {root}")
    clinical = pd.read_csv("/home/vicente/Github/Clear-Cell-Carcinoma-Study/Dataset/clinical_raw.csv", index_col=0)
    patient_ids = set(clinical.index)

    print(f"Unique Patients clinical: {len(patient_ids)}")
    #breakpoint()
    cptac_dir = root / "CT-Scan/CPTAC-CCRCC"

    present_dir = root / "CT-Scan/present2"
    not_present_dir = root / "CT-Scan/not_present2"
    present_dir.mkdir(exist_ok=True)
    not_present_dir.mkdir(exist_ok=True)
    
    total = 0
    for folder in cptac_dir.iterdir():
        if folder.is_dir():
            total += 1
            print(f"Processing folder: {folder.name} {total}/{len(list(cptac_dir.iterdir()))}")
            if folder.name in patient_ids:
                print(f"Moving {folder.name} to present2 directory")
                shutil.move(str(folder), str(present_dir / folder.name))
            else:
                print(f"Moving {folder.name} to not_present2 directory")
                shutil.move(str(folder), str(not_present_dir / folder.name))
    
    for folder in cptac_dir.iterdir():
        if folder.is_dir():
            for sub_folder in folder.iterdir():
                if "MR" in sub_folder.name and sub_folder.is_dir():
                    shutil.rmtree(sub_folder)

    num_present = len(list(present_dir.iterdir()))
    num_not_present = len(list(not_present_dir.iterdir()))
    print(f"Moved {num_present} folders to {present_dir}")
    print(f"Moved {num_not_present} folders to {not_present_dir}")



if __name__ == "__main__":
    main()
