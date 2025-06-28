import gdown
import subprocess
from pathlib import Path
import requests
import zipfile
import shutil
import argparse
from git import Repo
import os
from utils.download_kits23 import download_dataset
from natsort import natsorted


def setup_word_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir) / "ScribbleBench"
    archive_dir = dataset_dir / "archives"
    raw_dir = dataset_dir / "raw"
    preprocessed_dir = dataset_dir
    archive_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    ####################################################################################################################
    #### Download WORD dataset (no GT labels)
    ####################################################################################################################

    print("Downloading WORD dataset (no GT labels)...")
    url = 'https://drive.google.com/file/d/19OWCXZGrimafREhXm8O8w2HBHZTfxEgU/view'
    gdown.download(url, str(archive_dir / "WORD-V0.1.0.zip"), fuzzy=True)

    ####################################################################################################################
    #### Unpack WORD dataset archive
    ####################################################################################################################

    print("Unpacking WORD dataset archive...")
    subprocess.run([
        "7z", "x", archive_dir / "WORD-V0.1.0.zip",
        f"-pword@uestc",
        f"-o{raw_dir / "WORD"}"
    ], check=True)

    ####################################################################################################################
    #### Download WORD GT labels
    ####################################################################################################################

    print("Downloading WORD GT labels...")
    url = "https://github.com/HiLab-git/WORD/raw/main/WORD_V0.1.0_labelsTs.zip"
    response = requests.get(url)
    response.raise_for_status()  # Raise an error on bad status
    with open(archive_dir / "WORD_V0.1.0_labelsTs.zip", "wb") as f:
        f.write(response.content)

    ####################################################################################################################
    #### Unpack WORD labels archive
    ####################################################################################################################

    print("Unpacking WORD labels archive...")
    with zipfile.ZipFile(archive_dir / "WORD_V0.1.0_labelsTs.zip", 'r') as zip_ref:
        zip_ref.extractall(raw_dir / "WORD" / "WORD-V0.1.0")

    ####################################################################################################################
    #### Preprocess WORD dataset
    ####################################################################################################################

    print("Preprocessing WORD dataset...")
    word_raw_dir = raw_dir / "WORD" / "WORD-V0.1.0"
    word_preprocessed_dir = preprocessed_dir / "WORD"
    word_raw_dir = Path(word_raw_dir)
    word_preprocessed_dir = Path(word_preprocessed_dir)

    (word_preprocessed_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (word_preprocessed_dir / "imagesTs").mkdir(parents=True, exist_ok=True)
    (word_preprocessed_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    (word_preprocessed_dir / "labelsTs").mkdir(parents=True, exist_ok=True)

    names = [path.name[:-7] for path in (word_raw_dir / "imagesTr").rglob("*.nii.gz")]
    for name in names:
        shutil.move(word_raw_dir / "imagesTr" / f"{name}.nii.gz", word_preprocessed_dir / "imagesTr" / f"{name}_0000.nii.gz")

    names = [path.name[:-7] for path in (word_raw_dir / "imagesVal").rglob("*.nii.gz")]
    for name in names:
        shutil.move(word_raw_dir / "imagesVal" / f"{name}.nii.gz", word_preprocessed_dir / "imagesTr" / f"{name}_0000.nii.gz")

    names = [path.name[:-7] for path in (word_raw_dir / "imagesTs").rglob("*.nii.gz")]
    for name in names:
        shutil.move(word_raw_dir / "imagesTs" / f"{name}.nii.gz", word_preprocessed_dir / "imagesTs" / f"{name}_0000.nii.gz")


    names = [path.name[:-7] for path in (word_raw_dir / "labelsTr").rglob("*.nii.gz")]
    for name in names:
        shutil.move(word_raw_dir / "labelsTr" / f"{name}.nii.gz", word_preprocessed_dir / "labelsTr" / f"{name}.nii.gz")

    names = [path.name[:-7] for path in (word_raw_dir / "labelsVal").rglob("*.nii.gz")]
    for name in names:
        shutil.move(word_raw_dir / "labelsVal" / f"{name}.nii.gz", word_preprocessed_dir / "labelsTr" / f"{name}.nii.gz")

    names = [path.name[:-7] for path in (word_raw_dir / "labelsTs").rglob("*.nii.gz")]
    for name in names:
        shutil.move(word_raw_dir / "labelsTs" / f"{name}.nii.gz", word_preprocessed_dir / "labelsTs" / f"{name}.nii.gz")


    shutil.move(word_raw_dir / "dataset.json", word_preprocessed_dir / "dataset.json")

    ####################################################################################################################
    #### Delete archive and raw dataset files
    ####################################################################################################################

    print("Deleting archive and raw dataset files...")
    shutil.rmtree(archive_dir, ignore_errors=True)
    shutil.rmtree(raw_dir, ignore_errors=True)

    print("Finished setting up WORD dataset.")


def setup_mscmr_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir) / "ScribbleBench"
    archive_dir = dataset_dir / "archive"
    raw_dir = dataset_dir / "raw"
    preprocessed_dir = dataset_dir
    mscmr_preprocessed_dir = preprocessed_dir / "MSCMR"
    archive_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    mscmr_preprocessed_dir.mkdir(parents=True, exist_ok=True)

    ####################################################################################################################
    #### Download MSCMR dataset
    ####################################################################################################################

    print("Downloading MSCMR dataset...")
    repo_url = "https://github.com/BWGZK/CycleMix.git"
    repo_dir = raw_dir / "CycleMix"

    Repo.clone_from(repo_url, repo_dir)

    train_labels_url = "https://syncandshare.desy.de/index.php/s/j2t8g8P8LHb9Xfk/download/labelsTr.zip"
    response = requests.get(train_labels_url)
    response.raise_for_status()  # Raise an error on bad status
    with open(archive_dir / "labelsTr.zip", "wb") as f:
        f.write(response.content)

    ####################################################################################################################
    #### Unpack MSCMR labels archive
    ####################################################################################################################

    print("Unpacking MSCMR labels archive...")
    with zipfile.ZipFile(archive_dir / "labelsTr.zip", 'r') as zip_ref:
        zip_ref.extractall(mscmr_preprocessed_dir)

    ####################################################################################################################
    #### Preprocess MSCMR dataset
    ####################################################################################################################

    print("Preprocessing MSCMR dataset...")
    mscmr_raw_dir = repo_dir / "MSCMR_dataset"

    (mscmr_preprocessed_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (mscmr_preprocessed_dir / "imagesTs").mkdir(parents=True, exist_ok=True)
    (mscmr_preprocessed_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    (mscmr_preprocessed_dir / "labelsTs").mkdir(parents=True, exist_ok=True)

    names = [path.name[:-7] for path in (mscmr_raw_dir / "train" / "images").rglob("*.nii.gz")]
    for name in names:
        shutil.move(mscmr_raw_dir / "train" / "images" / f"{name}.nii.gz", mscmr_preprocessed_dir / "imagesTr" / f"{name}_0000.nii.gz")

    names = [path.name[:-7] for path in (mscmr_raw_dir / "val" / "images").rglob("*.nii.gz")]
    for name in names:
        shutil.move(mscmr_raw_dir / "val" / "images" / f"{name}.nii.gz", mscmr_preprocessed_dir / "imagesTr" / f"{name}_0000.nii.gz")

    names = [path.name[:-7] for path in (mscmr_raw_dir / "TestSet" / "images").rglob("*.nii.gz")]
    for name in names:
        shutil.move(mscmr_raw_dir / "TestSet" / "images" / f"{name}.nii.gz", mscmr_preprocessed_dir / "imagesTs" / f"{name}_0000.nii.gz")

    names = [path.name[:-7] for path in (mscmr_raw_dir / "TestSet" / "labels").rglob("*.nii.gz")]
    for name in names:
        shutil.move(mscmr_raw_dir / "TestSet" / "labels" / f"{name}.nii.gz", mscmr_preprocessed_dir / "labelsTs" / f"{name}.nii.gz")

    # These two images have no dense GT so it is not possible to generate scribbles for them
    os.remove(mscmr_preprocessed_dir / "imagesTr" / "subject2_DE_0000.nii.gz")
    os.remove(mscmr_preprocessed_dir / "imagesTr" / "subject4_DE_0000.nii.gz")

    dataset_json_url = "https://syncandshare.desy.de/index.php/s/9gdZ33WL2nPXpGC/download/dataset.json"
    response = requests.get(dataset_json_url)
    response.raise_for_status()  # Raise an error on bad status
    with open(mscmr_preprocessed_dir / "dataset.json", "wb") as f:
        f.write(response.content)

    ####################################################################################################################
    #### Delete archive and raw dataset files
    ####################################################################################################################

    print("Deleting archive and raw dataset files...")
    shutil.rmtree(archive_dir, ignore_errors=True)
    shutil.rmtree(raw_dir, ignore_errors=True)

    print("Finished setting up MSCMR dataset.")


def setup_kits_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir) / "ScribbleBench"
    raw_dir = dataset_dir / "raw"
    kits_raw_dir = raw_dir / "KiTS2023" / "dataset"
    preprocessed_dir = dataset_dir
    kits_preprocessed_dir = preprocessed_dir / "KiTS2023"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    kits_preprocessed_dir.mkdir(parents=True, exist_ok=True)

    test_set = ['case_00007', 'case_00013', 'case_00003', 'case_00018', 'case_00019', 'case_00016', 'case_00026', 'case_00014', 
                'case_00038', 'case_00000', 'case_00045', 'case_00050', 'case_00061', 'case_00070', 'case_00069', 'case_00074', 
                'case_00087', 'case_00090', 'case_00086', 'case_00084', 'case_00059', 'case_00106', 'case_00105', 'case_00102', 
                'case_00112', 'case_00117', 'case_00114', 'case_00121', 'case_00101', 'case_00096', 'case_00115', 'case_00126', 
                'case_00127', 'case_00120', 'case_00124', 'case_00116', 'case_00133', 'case_00145', 'case_00147', 'case_00152', 
                'case_00144', 'case_00118', 'case_00132', 'case_00135', 'case_00141', 'case_00146', 'case_00164', 'case_00167', 
                'case_00172', 'case_00179', 'case_00181', 'case_00184', 'case_00192', 'case_00194', 'case_00195', 'case_00165', 
                'case_00199', 'case_00210', 'case_00188', 'case_00191', 'case_00211', 'case_00212', 'case_00222', 'case_00217', 
                'case_00221', 'case_00227', 'case_00232', 'case_00236', 'case_00231', 'case_00234', 'case_00214', 'case_00238', 
                'case_00223', 'case_00237', 'case_00240', 'case_00216', 'case_00266', 'case_00269', 'case_00268', 'case_00275', 
                'case_00279', 'case_00253', 'case_00273', 'case_00282', 'case_00287', 'case_00286', 'case_00281', 'case_00284', 
                'case_00291', 'case_00283', 'case_00276', 'case_00404', 'case_00411', 'case_00416', 'case_00418', 'case_00422', 
                'case_00423', 'case_00290', 'case_00424', 'case_00295', 'case_00426', 'case_00428', 'case_00293', 'case_00429', 
                'case_00433', 'case_00441', 'case_00443', 'case_00437', 'case_00444', 'case_00452', 'case_00449', 'case_00453', 
                'case_00463', 'case_00468', 'case_00476', 'case_00483', 'case_00485', 'case_00480', 'case_00491', 'case_00474', 
                'case_00486', 'case_00496', 'case_00494', 'case_00492', 'case_00503', 'case_00442', 'case_00518', 'case_00521', 
                'case_00522', 'case_00525', 'case_00515', 'case_00533', 'case_00532', 'case_00539', 'case_00517', 'case_00546', 
                'case_00550', 'case_00554', 'case_00558', 'case_00557', 'case_00544', 'case_00567', 'case_00574', 'case_00555', 
                'case_00576', 'case_00575', 'case_00564']

    ####################################################################################################################
    #### Download KiTS2023 dataset
    ####################################################################################################################

    print("Downloading KiTS2023 dataset...")
    repo_url = "https://github.com/neheller/kits23.git"
    Repo.clone_from(repo_url, str(kits_raw_dir.parent))
    download_dataset(kits_raw_dir)

    ####################################################################################################################
    #### Preprocess KiTS2023 dataset
    ####################################################################################################################

    print("Preprocessing KiTS2023 dataset...")

    (kits_preprocessed_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (kits_preprocessed_dir / "imagesTs").mkdir(parents=True, exist_ok=True)
    (kits_preprocessed_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    (kits_preprocessed_dir / "labelsTs").mkdir(parents=True, exist_ok=True)

    names = [p.name for p in kits_raw_dir.iterdir() if p.is_dir()]
    names = natsorted(names)
    for name in names:
        postfix = "Tr" if name not in test_set else "Ts"
        shutil.move(kits_raw_dir / name / "imaging.nii.gz", kits_preprocessed_dir / f"images{postfix}" / f"{name}_0000.nii.gz")
        shutil.move(kits_raw_dir / name / "segmentation.nii.gz", kits_preprocessed_dir / f"labels{postfix}" / f"{name}.nii.gz")

    dataset_json_url = "https://syncandshare.desy.de/index.php/s/Cfpwyg5dmi9a2Df/download/dataset.json"
    response = requests.get(dataset_json_url)
    response.raise_for_status()  # Raise an error on bad status
    with open(kits_preprocessed_dir / "dataset.json", "wb") as f:
        f.write(response.content)

    ####################################################################################################################
    #### Delete raw dataset files
    ####################################################################################################################

    print("Deleting raw dataset files...")
    shutil.rmtree(raw_dir, ignore_errors=True)

    print("Finished setting up KiTS2023 dataset.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset_dir", required=True, type=str, help="Path to the dir used for setting up ScribbleBench.")
    parser.add_argument('--word', required=False, default=False, action="store_true", help="Download and preprocess the WORD dataset for ScribbleBench.")
    parser.add_argument('--mscmr', required=False, default=False, action="store_true", help="Download and preprocess the MSCMR dataset for ScribbleBench.")
    parser.add_argument('--kits', required=False, default=False, action="store_true", help="Download and preprocess the KiTS2023 dataset for ScribbleBench.")
    args = parser.parse_args()

    if args.word:
        setup_word_dataset(args.dataset_dir)
    if args.word:
        setup_mscmr_dataset(args.dataset_dir)
    if args.kits:
        setup_kits_dataset(args.dataset_dir)
        