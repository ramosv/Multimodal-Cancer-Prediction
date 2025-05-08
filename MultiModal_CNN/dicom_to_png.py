import dicom2jpg
from pathlib import Path
from PIL import Image
import numpy as np

def convert_to_png(dicom_dir):
    dicom_dir = Path(dicom_dir)
    if not dicom_dir.is_dir():
        raise ValueError(f"{dicom_dir} is not a directory")
    
    # nothing to return, this will conver the files and save them to a unique folder (timestamp)
    dicom2jpg.dicom2png(dicom_dir)

def check_blank_images(out_dir):
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise ValueError(f"{out_dir} is not a directory")
    
    for folder in out_dir.iterdir():
        for file in folder.glob("*.png"):
            img = Image.open(file)
            arr = np.array(img)
            if arr.max() == arr.min():
                print(f"Possibly blank image: {file.name}")

def convert_single_dicom(dicom_img):
    dicom_img = Path(dicom_img)
    if not dicom_img.is_file():
        raise ValueError(f"{dicom_img} is not a file")
    
    dicom2jpg.dicom2png(dicom_img)

# if __name__ == "__main__":
#     root = Path.cwd()

#     dicom_img_01 = root / r"CT-Scan\CPTAC-CCRCC\C3L-00608\03-10-2012-NA-CT ABDOMEN WITH AND WITHOUT CONTRAST-69573\1.000000-SCOUT-99132\1-1.dcm"
#     convert_single_dicom(dicom_img_01)

#     covert_to_png(root / "CT-Scan/present_png")
#     convert_to_png(root / "CT-Scan/not_present_png")

#     check_blank_images(root / "CT-Scan/present_png")
#     check_blank_images(root / "CT-Scan/not_present_png")