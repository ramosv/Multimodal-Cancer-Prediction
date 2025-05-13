from pathlib import Path
import pandas as pd
import shutil
import os
import logging
import sys
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream = sys.stdout)


def find_overlap(clinical_df, image_dir):
    patient_ids = set(clinical_df.index)
    logging.info(f"Unique Patients clinical: {len(patient_ids)}")

    # we removed MRI images from the folders to avoid confusion
    mr_count = 0
    for folder in image_dir.iterdir():
        if folder.is_dir():
            for sub_folder in folder.iterdir():
                if "MR" in sub_folder.name and sub_folder.is_dir():
                    mr_count += 1
                    shutil.rmtree(sub_folder)
                    logging.info(f"Removed {sub_folder.name} from {folder.name}")

    logging.info(f"Removed {mr_count} MR folders from the dataset")

    
    present_dir = Path("CT-Scan/present")
    not_present_dir = Path("CT-Scan/not_present")
    os.makedirs(present_dir, exist_ok=True)
    os.makedirs(not_present_dir, exist_ok=True)
    logging.info(f"Present directory: {present_dir}")
    logging.info(f"Not present directory: {not_present_dir}")
    
    total = 0
    for folder in image_dir.iterdir():
        if folder.is_dir():
            total += 1
            logging.info(f"\nProcessing folder: {folder.name} {total}/{len(list(image_dir.iterdir()))}")
            if folder.name in patient_ids:
                logging.info(f"Moving {folder.name} to present directory\n")
                shutil.copytree(str(folder), str(present_dir / folder.name), dirs_exist_ok=True)
            else:
                logging.info(f"Moving {folder.name} to not_present directory\n")
                shutil.copytree(str(folder), str(not_present_dir / folder.name), dirs_exist_ok=True)
    num_present = len(list(present_dir.iterdir()))
    num_not_present = len(list(not_present_dir.iterdir()))

    logging.info(f"Moved {num_present} folders from {image_dir} to {present_dir}")
    logging.info(f"Moved {num_not_present} folders from {image_dir} to {not_present_dir}")

# if __name__ == "__main__":
#     root = Path.cwd()
#     clinical = pd.read_csv(root / "CT-Scan/clinical.csv", index_col=0)
#     images = root / "CT-Scan/CPTAC-CCRCC"

#     find_overlap(clinical, images)

