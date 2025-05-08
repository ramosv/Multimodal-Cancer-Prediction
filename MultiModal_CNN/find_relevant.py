from pathlib import Path
import pandas as pd
import shutil

def find_overlap(clinical_df, image_dir):
    patient_ids = set(clinical_df.index)
    print(f"Unique Patients clinical: {len(patient_ids)}")

    # we removed MRI images from the folders to avoid confusion
    mr_count = 0
    for folder in image_dir.iterdir():
        if folder.is_dir():
            for sub_folder in folder.iterdir():
                if "MR" in sub_folder.name and sub_folder.is_dir():
                    mr_count += 1
                    shutil.rmtree(sub_folder)
                    print(f"Removed {sub_folder.name} from {folder.name}")

    print(f"Removed {mr_count} MR folders from the dataset")

    
    present_dir = "CT-Scan/present_png"
    not_present_dir = "CT-Scan/not_present_png"
    present_dir.mkdir(exist_ok=True)
    not_present_dir.mkdir(exist_ok=True)
    print(f"Present directory: {present_dir}")
    print(f"Not present directory: {not_present_dir}")
    
    total = 0
    for folder in image_dir.iterdir():
        if folder.is_dir():
            total += 1
            print(f"Processing folder: {folder.name} {total}/{len(list(image_dir.iterdir()))}")
            if folder.name in patient_ids:
                print(f"Moving {folder.name} to present directory")
                shutil.move(str(folder), str(present_dir / folder.name))
            else:
                print(f"Moving {folder.name} to not_present directory")
                shutil.move(str(folder), str(not_present_dir / folder.name))

    num_present = len(list(present_dir.iterdir()))
    num_not_present = len(list(not_present_dir.iterdir()))

    print(f"Moved {num_present} folders from {image_dir} to {present_dir}")
    print(f"Moved {num_not_present} folders from {image_dir} to {not_present_dir}")

# if __name__ == "__main__":
#     root = Path.cwd()
#     clinical = pd.read_csv(root / "CT-Scan/clinical.csv", index_col=0)
#     images = root / "CT-Scan/CPTAC-CCRCC"

#     find_overlap(clinical, images)

