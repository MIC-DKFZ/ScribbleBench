import gdown
import subprocess
from pathlib import Path
import requests
import zipfile
import shutil
import argparse
from git import Repo
import os


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
    #### Preprocess WORD dataset
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
        shutil.move(mscmr_raw_dir / "TestSet" / "labels" / f"{name}.nii.gz", mscmr_preprocessed_dir / "labelsTs" / f"{name}_0000.nii.gz")

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset_dir", required=True, type=str, help="Path to the dir used for setting up ScribbleBench.")
    parser.add_argument('--word', required=False, default=False, action="store_true", help="Download and preprocess the WORD dataset for ScribbleBench.")
    parser.add_argument('--mscmr', required=False, default=False, action="store_true", help="Download and preprocess the MSCMR dataset for ScribbleBench.")
    args = parser.parse_args()

    if args.word:
        setup_word_dataset(args.dataset_dir)
    if args.word:
        setup_mscmr_dataset(args.dataset_dir)
