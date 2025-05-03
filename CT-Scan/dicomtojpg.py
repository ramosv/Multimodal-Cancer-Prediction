import dicom2jpg
from pathlib import Path


if __name__ == "__main__":

    # dicom_img_01 = Path(r"C:\Users\ramos\Desktop\GitHub\Clear-Cell-Carcinoma-Study\CT-Scan\CPTAC-CCRCC\C3L-00608\03-10-2012-NA-CT ABDOMEN WITH AND WITHOUT CONTRAST-69573\1.000000-SCOUT-99132\1-1.dcm")
    # dicom2jpg.dicom2jpg(dicom_img_01)
    # root = Path.cwd()
    #dicom_dir = root / "CT-Scan/present"
    # dicom_dir2 = root / "CT-Scan/not_present"
    # dicom_dir = Path(r"C:\Users\ramos\Desktop\GitHub\Clear-Cell-Carcinoma-Study\CT-Scan\present")
    # dicom2jpg.dicom2png(dicom_dir2)
    #dicom2jpg.dicom2png(dicom_dir2)

    from PIL import Image
    import numpy as np
    from pathlib import Path

    out_dir = Path("/Users/jessicatan/CU Denver 2024/Spring 2025/CSCI 5930/Research Project/Clear-Cell-Carcinoma-Study/CT-Scan/present_png")
    for folder in out_dir.iterdir():
        for file in folder.glob("*.png"):
            img = Image.open(file)
            arr = np.array(img)
            if arr.max() == arr.min():
                print(f"Possibly blank image: {file.name}")
    
#export_location = "/Users/user/Desktop/BMP_files"

# convert single DICOM file to jpg format


# convert all DICOM files in dicom_dir folder to png format
#dicom2jpg.dicom2png(dicom_dir)  

# # convert all DICOM files in dicom_dir folder to bmp, to a specified location
# dicom2jpg.dicom2bmp(dicom_dir, target_root=export_location) 

# # convert single DICOM file to numpy.ndarray for further use
# img_data = dicom2jpg.dicom2img(dicom_img_01)

# # convert DICOM ByteIO to numpy.ndarray
# img_data = dicom2jpg.io2img(dicomIO)