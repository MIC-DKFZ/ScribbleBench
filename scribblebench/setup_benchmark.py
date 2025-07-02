import gdown
import subprocess
from pathlib import Path
import zipfile
import shutil
import argparse
from git import Repo
import os
from utils.download_kits23 import download_dataset
from utils.utils import info2dict, download
from natsort import natsorted
import tarfile


def setup_word_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir) / "ScribbleBench"
    archive_dir = dataset_dir / "archives"
    raw_dir = dataset_dir / "raw"
    preprocessed_dir = dataset_dir
    archive_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    ####################################################################################################################
    #### Download WORD dataset
    ####################################################################################################################

    print("Downloading WORD dataset...")
    url = 'https://drive.google.com/file/d/19OWCXZGrimafREhXm8O8w2HBHZTfxEgU/view'
    gdown.download(url, str(archive_dir / "WORD-V0.1.0.zip"), fuzzy=True)

    url = "https://github.com/HiLab-git/WORD/raw/main/WORD_V0.1.0_labelsTs.zip"
    download(url, archive_dir / "WORD_V0.1.0_labelsTs.zip")

    scribbles_url = "https://syncandshare.desy.de/index.php/s/DJ4KBZrZScFbTei/download/ScribbleBench_scribbles.zip"
    download(scribbles_url, archive_dir / "ScribbleBench_scribbles.zip")

    dataset_json_url = "https://syncandshare.desy.de/index.php/s/CsEzWewcxpkoC55/download/dataset.json"
    download(dataset_json_url, word_preprocessed_dir / "dataset.json")

    ####################################################################################################################
    #### Unpack WORD dataset archive
    ####################################################################################################################

    print("Unpacking WORD dataset archive...")
    subprocess.run([
        "7z", "x", archive_dir / "WORD-V0.1.0.zip",
        f"-pword@uestc",
        f"-o{raw_dir / "WORD"}"
    ], check=True)

    with zipfile.ZipFile(archive_dir / "WORD_V0.1.0_labelsTs.zip", 'r') as zip_ref:
        zip_ref.extractall(raw_dir / "WORD" / "WORD-V0.1.0")

    with zipfile.ZipFile(archive_dir / "ScribbleBench_scribbles.zip", 'r') as zip_ref:
        zip_ref.extractall(raw_dir / "WORD")

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
    (word_preprocessed_dir / "labelsTr_dense").mkdir(parents=True, exist_ok=True)
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
        shutil.move(word_raw_dir / "labelsTr" / f"{name}.nii.gz", word_preprocessed_dir / "labelsTr_dense" / f"{name}.nii.gz")

    names = [path.name[:-7] for path in (word_raw_dir / "labelsVal").rglob("*.nii.gz")]
    for name in names:
        shutil.move(word_raw_dir / "labelsVal" / f"{name}.nii.gz", word_preprocessed_dir / "labelsTr_dense" / f"{name}.nii.gz")

    names = [path.name[:-7] for path in (word_raw_dir / "labelsTs").rglob("*.nii.gz")]
    for name in names:
        shutil.move(word_raw_dir / "labelsTs" / f"{name}.nii.gz", word_preprocessed_dir / "labelsTs" / f"{name}.nii.gz")

    names = [path.name[:-7] for path in (raw_dir / "WORD" / "ScribbleBench_scribbles" / "WORD" / "scribblesTr").rglob("*.nii.gz")]
    for name in names:
        shutil.move(raw_dir / "WORD" / "ScribbleBench_scribbles" / "WORD" / "scribblesTr" / f"{name}.nii.gz", word_preprocessed_dir / "labelsTr" / f"{name}.nii.gz")

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
    download(train_labels_url, archive_dir / "labelsTr.zip")

    scribbles_url = "https://syncandshare.desy.de/index.php/s/DJ4KBZrZScFbTei/download/ScribbleBench_scribbles.zip"
    download(scribbles_url, archive_dir / "ScribbleBench_scribbles.zip")

    dataset_json_url = "https://syncandshare.desy.de/index.php/s/9gdZ33WL2nPXpGC/download/dataset.json"
    download(dataset_json_url, mscmr_preprocessed_dir / "dataset.json")

    ####################################################################################################################
    #### Unpack MSCMR labels archive
    ####################################################################################################################

    print("Unpacking MSCMR labels archive...")
    with zipfile.ZipFile(archive_dir / "labelsTr.zip", 'r') as zip_ref:
        zip_ref.extractall(raw_dir)

    with zipfile.ZipFile(archive_dir / "ScribbleBench_scribbles.zip", 'r') as zip_ref:
        zip_ref.extractall(raw_dir)

    ####################################################################################################################
    #### Preprocess MSCMR dataset
    ####################################################################################################################

    print("Preprocessing MSCMR dataset...")
    mscmr_raw_dir = repo_dir / "MSCMR_dataset"

    (mscmr_preprocessed_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (mscmr_preprocessed_dir / "imagesTs").mkdir(parents=True, exist_ok=True)
    (mscmr_preprocessed_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    (mscmr_preprocessed_dir / "labelsTr_dense").mkdir(parents=True, exist_ok=True)
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

    names = [path.name[:-7] for path in (raw_dir / "labelsTr").rglob("*.nii.gz")]
    for name in names:
        shutil.move(raw_dir / "labelsTr" / f"{name}.nii.gz", mscmr_preprocessed_dir / "labelsTr_dense" / f"{name}.nii.gz")

    names = [path.name[:-7] for path in (mscmr_raw_dir / "TestSet" / "labels").rglob("*.nii.gz")]
    for name in names:
        shutil.move(mscmr_raw_dir / "TestSet" / "labels" / f"{name}.nii.gz", mscmr_preprocessed_dir / "labelsTs" / f"{name}.nii.gz")

    names = [path.name[:-7] for path in (raw_dir / "ScribbleBench_scribbles" / "MSCMR" / "scribblesTr").rglob("*.nii.gz")]
    for name in names:
        shutil.move(raw_dir / "ScribbleBench_scribbles" / "MSCMR" / "scribblesTr" / f"{name}.nii.gz", mscmr_preprocessed_dir / "labelsTr" / f"{name}.nii.gz")

    # These two images have no dense GT so it is not possible to generate scribbles for them
    os.remove(mscmr_preprocessed_dir / "imagesTr" / "subject2_DE_0000.nii.gz")
    os.remove(mscmr_preprocessed_dir / "imagesTr" / "subject4_DE_0000.nii.gz")

    ####################################################################################################################
    #### Delete archive and raw dataset files
    ####################################################################################################################

    print("Deleting archive and raw dataset files...")
    shutil.rmtree(archive_dir, ignore_errors=True)
    shutil.rmtree(raw_dir, ignore_errors=True)

    print("Finished setting up MSCMR dataset.")


def setup_kits_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir) / "ScribbleBench"
    archive_dir = dataset_dir / "archive"
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

    scribbles_url = "https://syncandshare.desy.de/index.php/s/DJ4KBZrZScFbTei/download/ScribbleBench_scribbles.zip"
    download(scribbles_url, archive_dir / "ScribbleBench_scribbles.zip")

    dataset_json_url = "https://syncandshare.desy.de/index.php/s/Cfpwyg5dmi9a2Df/download/dataset.json"
    download(dataset_json_url, kits_preprocessed_dir / "dataset.json")

    ####################################################################################################################
    #### Unpack KiTS2023 dataset archive
    ####################################################################################################################

    print("Unpacking KiTS2023 dataset archive...")

    with zipfile.ZipFile(archive_dir / "ScribbleBench_scribbles.zip", 'r') as zip_ref:
        zip_ref.extractall(raw_dir)

    ####################################################################################################################
    #### Preprocess KiTS2023 dataset
    ####################################################################################################################

    print("Preprocessing KiTS2023 dataset...")

    (kits_preprocessed_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (kits_preprocessed_dir / "imagesTs").mkdir(parents=True, exist_ok=True)
    (kits_preprocessed_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    (kits_preprocessed_dir / "labelsTr_dense").mkdir(parents=True, exist_ok=True)
    (kits_preprocessed_dir / "labelsTs").mkdir(parents=True, exist_ok=True)

    names = [p.name for p in kits_raw_dir.iterdir() if p.is_dir()]
    names = natsorted(names)
    for name in names:
        postfix = "Tr" if name not in test_set else "Ts"
        shutil.move(kits_raw_dir / name / "imaging.nii.gz", kits_preprocessed_dir / f"images{postfix}" / f"{name}_0000.nii.gz")
        if postfix == "Tr":
            shutil.move(kits_raw_dir / name / "segmentation.nii.gz", kits_preprocessed_dir / f"labels{postfix}_dense" / f"{name}.nii.gz")
        else:
            shutil.move(kits_raw_dir / name / "segmentation.nii.gz", kits_preprocessed_dir / f"labels{postfix}" / f"{name}.nii.gz")

    names = [path.name[:-7] for path in (raw_dir / "ScribbleBench_scribbles" / "KiTS2023" / "scribblesTr").rglob("*.nii.gz")]
    for name in names:
        shutil.move(raw_dir / "ScribbleBench_scribbles" / "KiTS2023" / "scribblesTr" / f"{name}.nii.gz", kits_preprocessed_dir / "labelsTr" / f"{name}.nii.gz")

    ####################################################################################################################
    #### Delete raw dataset files
    ####################################################################################################################

    print("Deleting raw dataset files...")
    shutil.rmtree(raw_dir, ignore_errors=True)

    print("Finished setting up KiTS2023 dataset.")


def setup_lits_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir) / "ScribbleBench"
    archive_dir = dataset_dir / "archive"
    raw_dir = dataset_dir / "raw"
    lits_raw_dir = raw_dir / "Task03_Liver"
    preprocessed_dir = dataset_dir
    lits_preprocessed_dir = preprocessed_dir / "LiTS"
    archive_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    lits_preprocessed_dir.mkdir(parents=True, exist_ok=True)

    test_set = ['liver_112', 'liver_38', 'liver_47', 'liver_84', 'liver_119', 'liver_15', 'liver_62', 'liver_72', 'liver_98', 
                'liver_0', 'liver_66', 'liver_7', 'liver_58', 'liver_93', 'liver_104', 'liver_43', 'liver_64', 'liver_91', 'liver_126', 
                'liver_69', 'liver_102', 'liver_123', 'liver_127', 'liver_26', 'liver_35', 'liver_45', 'liver_73', 'liver_89', 
                'liver_106', 'liver_29', 'liver_36', 'liver_109', 'liver_12', 'liver_128', 'liver_20', 'liver_54', 'liver_8', 
                'liver_19', 'liver_21']

    ####################################################################################################################
    #### Download LiTS dataset
    ####################################################################################################################

    # Instead of the LiTS we are downloading the MSD Task03_Liver dataset, which is the exact same as the LiTS dataset but with fixed labels
    print("Downloading LiTS dataset...")
    url = 'https://drive.google.com/file/d/1jyVGUGyxKBXV6_9ivuZapQS8eUJXCIpu/view'
    gdown.download(url, str(archive_dir / "Task03_Liver.tar"), fuzzy=True)

    scribbles_url = "https://syncandshare.desy.de/index.php/s/DJ4KBZrZScFbTei/download/ScribbleBench_scribbles.zip"
    download(scribbles_url, archive_dir / "ScribbleBench_scribbles.zip")

    dataset_json_url = 'https://syncandshare.desy.de/index.php/s/wBMtJcFm6D2icXA/download/dataset.json'
    download(dataset_json_url, lits_preprocessed_dir / "dataset.json")

    ####################################################################################################################
    #### Unpack LiTS archive
    ####################################################################################################################

    print("Unpacking LiTS archive...")
    with tarfile.open(archive_dir / "Task03_Liver.tar", "r:*") as tar:
        tar.extractall(path=raw_dir)

    with zipfile.ZipFile(archive_dir / "ScribbleBench_scribbles.zip", 'r') as zip_ref:
        zip_ref.extractall(raw_dir)

    ####################################################################################################################
    #### Preprocess LiTS dataset
    ####################################################################################################################

    print("Preprocessing LiTS dataset...")

    (lits_preprocessed_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (lits_preprocessed_dir / "imagesTs").mkdir(parents=True, exist_ok=True)
    (lits_preprocessed_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    (lits_preprocessed_dir / "labelsTr_dense").mkdir(parents=True, exist_ok=True)
    (lits_preprocessed_dir / "labelsTs").mkdir(parents=True, exist_ok=True)

    names = [p.name[:-7] for p in (lits_raw_dir / "labelsTr").iterdir()]
    names = natsorted(names)
    for name in names:
        postfix = "Tr" if name not in test_set else "Ts"
        shutil.move(lits_raw_dir / "imagesTr" / f"{name}.nii.gz", lits_preprocessed_dir / f"images{postfix}" / f"{name}_0000.nii.gz")
        if postfix == "Tr":
            shutil.move(lits_raw_dir / "labelsTr" / f"{name}.nii.gz", lits_preprocessed_dir / f"labels{postfix}_dense" / f"{name}.nii.gz")
        else:
            shutil.move(lits_raw_dir / "labelsTr" / f"{name}.nii.gz", lits_preprocessed_dir / f"labels{postfix}" / f"{name}.nii.gz")

    names = [path.name[:-7] for path in (raw_dir / "ScribbleBench_scribbles" / "LiTS" / "scribblesTr").rglob("*.nii.gz")]
    for name in names:
        shutil.move(raw_dir / "ScribbleBench_scribbles" / "LiTS" / "scribblesTr" / f"{name}.nii.gz", lits_preprocessed_dir / "labelsTr" / f"{name}.nii.gz")

    ####################################################################################################################
    #### Delete raw dataset files
    ####################################################################################################################

    print("Deleting archive and raw dataset files...")
    shutil.rmtree(archive_dir, ignore_errors=True)
    shutil.rmtree(raw_dir, ignore_errors=True)

    print("Finished setting up LiTS dataset.")


def setup_acdc_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir) / "ScribbleBench"
    archive_dir = dataset_dir / "archive"
    raw_dir = dataset_dir / "raw"
    acdc_raw_dir = raw_dir
    preprocessed_dir = dataset_dir
    acdc_preprocessed_dir = preprocessed_dir / "ACDC"
    archive_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    acdc_preprocessed_dir.mkdir(parents=True, exist_ok=True)

    test_set = ['patient072_ED', 'patient041_ED', 'patient078_ED', 'patient024_ED', 'patient060_ES', 'patient078_ES', 'patient073_ED', 
                'patient082_ED', 'patient010_ED', 'patient005_ED', 'patient077_ED', 'patient080_ED', 'patient024_ES', 'patient077_ES', 
                'patient030_ES', 'patient039_ES', 'patient041_ES', 'patient082_ES', 'patient060_ED', 'patient010_ES', 'patient030_ED', 
                'patient005_ES', 'patient036_ES', 'patient073_ES', 'patient064_ES', 'patient039_ED', 'patient080_ES', 'patient064_ED', 
                'patient036_ED', 'patient072_ES']

    ####################################################################################################################
    #### Download ACDC dataset
    ####################################################################################################################

    print("Downloading ACDC dataset...")

    url = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/collection/637218c173e9f0047faa00fb/download"
    acdc_archive_file = archive_dir / "ACDC.zip"
    download(url, acdc_archive_file, int(2452590457))

    scribbles_url = "https://syncandshare.desy.de/index.php/s/DJ4KBZrZScFbTei/download/ScribbleBench_scribbles.zip"
    download(scribbles_url, archive_dir / "ScribbleBench_scribbles.zip")

    dataset_json_url = "https://syncandshare.desy.de/index.php/s/KCDbLyeMwwZpFH5/download/dataset.json"
    download(dataset_json_url, acdc_preprocessed_dir / "dataset.json")

    ####################################################################################################################
    #### Unpack ACDC archive
    ####################################################################################################################

    print("Unpacking ACDC archive...")
    with zipfile.ZipFile(acdc_archive_file, 'r') as zip_ref:
        zip_ref.extractall(acdc_raw_dir)

    with zipfile.ZipFile(archive_dir / "ScribbleBench_scribbles.zip", 'r') as zip_ref:
        zip_ref.extractall(raw_dir)

    ####################################################################################################################
    #### Preprocess ACDC dataset
    ####################################################################################################################

    print("Preprocessing ACDC dataset...")

    (acdc_preprocessed_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (acdc_preprocessed_dir / "imagesTs").mkdir(parents=True, exist_ok=True)
    (acdc_preprocessed_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    (acdc_preprocessed_dir / "labelsTr_dense").mkdir(parents=True, exist_ok=True)
    (acdc_preprocessed_dir / "labelsTs").mkdir(parents=True, exist_ok=True)

    acdc_train_raw_dir = acdc_raw_dir / "ACDC" / "database" / "training"
    names = [p.name for p in acdc_train_raw_dir.iterdir() if p.is_dir()]
    names = natsorted(names)

    for name in names:
        info = info2dict(acdc_train_raw_dir / name / "Info.cfg")
        ed_name = f"{name}_frame{str(info["ED"]).zfill(2)}"
        es_name = f"{name}_frame{str(info["ES"]).zfill(2)}"
        postfix = "Tr" if f"{name}_ED" not in test_set else "Ts"
        shutil.move(acdc_train_raw_dir / name / f"{ed_name}.nii.gz", acdc_preprocessed_dir / f"images{postfix}" / f"{name}_ED_0000.nii.gz")
        shutil.move(acdc_train_raw_dir / name / f"{es_name}.nii.gz", acdc_preprocessed_dir / f"images{postfix}" / f"{name}_ES_0000.nii.gz")
        if postfix == "Tr":
            shutil.move(acdc_train_raw_dir / name / f"{ed_name}_gt.nii.gz", acdc_preprocessed_dir / f"labels{postfix}_dense" / f"{name}_ED.nii.gz")
            shutil.move(acdc_train_raw_dir / name / f"{es_name}_gt.nii.gz", acdc_preprocessed_dir / f"labels{postfix}_dense" / f"{name}_ES.nii.gz")
        else:
            shutil.move(acdc_train_raw_dir / name / f"{ed_name}_gt.nii.gz", acdc_preprocessed_dir / f"labels{postfix}" / f"{name}_ED.nii.gz")
            shutil.move(acdc_train_raw_dir / name / f"{es_name}_gt.nii.gz", acdc_preprocessed_dir / f"labels{postfix}" / f"{name}_ES.nii.gz")

    names = [path.name[:-7] for path in (raw_dir / "ScribbleBench_scribbles" / "ACDC" / "scribblesTr").rglob("*.nii.gz")]
    for name in names:
        shutil.move(raw_dir / "ScribbleBench_scribbles" / "ACDC" / "scribblesTr" / f"{name}.nii.gz", acdc_preprocessed_dir / "labelsTr" / f"{name}.nii.gz")

    ####################################################################################################################
    #### Delete raw dataset files
    ####################################################################################################################

    print("Deleting archive and raw dataset files...")
    shutil.rmtree(archive_dir, ignore_errors=True)
    shutil.rmtree(raw_dir, ignore_errors=True)

    print("Finished setting up ACDC dataset.")


def setup_amos_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir) / "ScribbleBench"
    archive_dir = dataset_dir / "archive"
    raw_dir = dataset_dir / "raw"
    amos_raw_dir = raw_dir / "amos22"
    preprocessed_dir = dataset_dir
    amos_preprocessed_dir = preprocessed_dir / "AMOS2022_task2"
    archive_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    amos_preprocessed_dir.mkdir(parents=True, exist_ok=True)

    test_set = ['amos_0231', 'amos_0027', 'amos_0585', 'amos_0571', 'amos_0009', 'amos_0404', 'amos_0089', 'amos_0532', 'amos_0044', 
                'amos_0320', 'amos_0580', 'amos_0005', 'amos_0071', 'amos_0403', 'amos_0522', 'amos_0510', 'amos_0162', 'amos_0048', 
                'amos_0376', 'amos_0281', 'amos_0064', 'amos_0115', 'amos_0192', 'amos_0076', 'amos_0153', 'amos_0401', 'amos_0160', 
                'amos_0104', 'amos_0186', 'amos_0299', 'amos_0181', 'amos_0530', 'amos_0371', 'amos_0408', 'amos_0400', 'amos_0554', 
                'amos_0600', 'amos_0050', 'amos_0180', 'amos_0248', 'amos_0358', 'amos_0035', 'amos_0557', 'amos_0317', 'amos_0116', 
                'amos_0332', 'amos_0113', 'amos_0294', 'amos_0110', 'amos_0297', 'amos_0075', 'amos_0118', 'amos_0217', 'amos_0015', 
                'amos_0508', 'amos_0125', 'amos_0596', 'amos_0301', 'amos_0578', 'amos_0215', 'amos_0379', 'amos_0006', 'amos_0078', 
                'amos_0226', 'amos_0263', 'amos_0538', 'amos_0336', 'amos_0370', 'amos_0052', 'amos_0274', 'amos_0047', 'amos_0121']

    ####################################################################################################################
    #### Download AMOS2022_task2 dataset
    ####################################################################################################################

    dataset_url = "https://zenodo.org/records/7155725/files/amos22.zip?download=1"
    download(dataset_url, archive_dir / "amos22.zip", int(24.2 * 1024**3))

    scribbles_url = "https://syncandshare.desy.de/index.php/s/DJ4KBZrZScFbTei/download/ScribbleBench_scribbles.zip"
    download(scribbles_url, archive_dir / "ScribbleBench_scribbles.zip")

    dataset_json_url = "https://syncandshare.desy.de/index.php/s/kWMmc2ggXpjDFnJ/download/dataset.json"
    download(dataset_json_url, amos_preprocessed_dir / "dataset.json")

    ####################################################################################################################
    #### Unpack AMOS2022_task2 dataset archive
    ####################################################################################################################

    print("Unpacking AMOS2022_task2 dataset archive...")

    with zipfile.ZipFile(archive_dir / "amos22.zip", 'r') as zip_ref:
        zip_ref.extractall(raw_dir)

    with zipfile.ZipFile(archive_dir / "ScribbleBench_scribbles.zip", 'r') as zip_ref:
        zip_ref.extractall(raw_dir)

    ####################################################################################################################
    #### Preprocess AMOS2022_task2 dataset
    ####################################################################################################################

    print("Preprocessing AMOS2022_task2 dataset...")

    (amos_preprocessed_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (amos_preprocessed_dir / "imagesTs").mkdir(parents=True, exist_ok=True)
    (amos_preprocessed_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    (amos_preprocessed_dir / "labelsTr_dense").mkdir(parents=True, exist_ok=True)
    (amos_preprocessed_dir / "labelsTs").mkdir(parents=True, exist_ok=True)

    names = [p.name[:-7] for p in (amos_raw_dir / "labelsTr").iterdir()]
    names = natsorted(names)

    for name in names:
        postfix = "Tr" if name not in test_set else "Ts"
        shutil.move(amos_raw_dir / "imagesTr" / f"{name}.nii.gz", amos_preprocessed_dir / f"images{postfix}" / f"{name}_0000.nii.gz")
        if postfix == "Tr":
            shutil.move(amos_raw_dir / "labelsTr" / f"{name}.nii.gz", amos_preprocessed_dir / f"labels{postfix}_dense" / f"{name}.nii.gz")
        else:
            shutil.move(amos_raw_dir / "labelsTr" / f"{name}.nii.gz", amos_preprocessed_dir / f"labels{postfix}" / f"{name}.nii.gz")

    names = [path.name[:-7] for path in (raw_dir / "ScribbleBench_scribbles" / "AMOS2022_task2" / "scribblesTr").rglob("*.nii.gz")]
    for name in names:
        shutil.move(raw_dir / "ScribbleBench_scribbles" / "AMOS2022_task2" / "scribblesTr" / f"{name}.nii.gz", amos_preprocessed_dir / "labelsTr" / f"{name}.nii.gz")

    ####################################################################################################################
    #### Delete raw dataset files
    ####################################################################################################################

    print("Deleting raw dataset files...")
    shutil.rmtree(archive_dir, ignore_errors=True)
    shutil.rmtree(raw_dir, ignore_errors=True)

    print("Finished setting up AMOS2022_task2 dataset.")


def setup_brats_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir) / "ScribbleBench"
    archive_dir = dataset_dir / "archive"
    raw_dir = dataset_dir / "raw"
    brats_raw_dir = raw_dir / "BraTS_2020"
    preprocessed_dir = dataset_dir
    brats_preprocessed_dir = preprocessed_dir / "BraTS2020"
    archive_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    brats_raw_dir.mkdir(parents=True, exist_ok=True)
    brats_preprocessed_dir.mkdir(parents=True, exist_ok=True)

    test_set = ['BraTS20_Training_263', 'BraTS20_Training_119', 'BraTS20_Training_107', 'BraTS20_Training_173', 'BraTS20_Training_282', 
                'BraTS20_Training_296', 'BraTS20_Training_007', 'BraTS20_Training_272', 'BraTS20_Training_299', 'BraTS20_Training_368', 
                'BraTS20_Training_218', 'BraTS20_Training_280', 'BraTS20_Training_155', 'BraTS20_Training_180', 'BraTS20_Training_015', 
                'BraTS20_Training_019', 'BraTS20_Training_152', 'BraTS20_Training_206', 'BraTS20_Training_264', 'BraTS20_Training_101', 
                'BraTS20_Training_326', 'BraTS20_Training_269', 'BraTS20_Training_067', 'BraTS20_Training_188', 'BraTS20_Training_304', 
                'BraTS20_Training_278', 'BraTS20_Training_142', 'BraTS20_Training_309', 'BraTS20_Training_317', 'BraTS20_Training_124', 
                'BraTS20_Training_072', 'BraTS20_Training_226', 'BraTS20_Training_364', 'BraTS20_Training_047', 'BraTS20_Training_291', 
                'BraTS20_Training_325', 'BraTS20_Training_071', 'BraTS20_Training_323', 'BraTS20_Training_024', 'BraTS20_Training_248', 
                'BraTS20_Training_143', 'BraTS20_Training_018', 'BraTS20_Training_207', 'BraTS20_Training_059', 'BraTS20_Training_038', 
                'BraTS20_Training_041', 'BraTS20_Training_008', 'BraTS20_Training_344', 'BraTS20_Training_311', 'BraTS20_Training_262', 
                'BraTS20_Training_361', 'BraTS20_Training_252', 'BraTS20_Training_037', 'BraTS20_Training_250', 'BraTS20_Training_125', 
                'BraTS20_Training_283', 'BraTS20_Training_085', 'BraTS20_Training_165', 'BraTS20_Training_227', 'BraTS20_Training_231', 
                'BraTS20_Training_193', 'BraTS20_Training_338', 'BraTS20_Training_137', 'BraTS20_Training_298', 'BraTS20_Training_214', 
                'BraTS20_Training_044', 'BraTS20_Training_318', 'BraTS20_Training_321', 'BraTS20_Training_281', 'BraTS20_Training_081', 
                'BraTS20_Training_098', 'BraTS20_Training_301', 'BraTS20_Training_198', 'BraTS20_Training_070', 'BraTS20_Training_354', 
                'BraTS20_Training_001', 'BraTS20_Training_315', 'BraTS20_Training_184', 'BraTS20_Training_179', 'BraTS20_Training_009', 
                'BraTS20_Training_251', 'BraTS20_Training_118', 'BraTS20_Training_075', 'BraTS20_Training_172', 'BraTS20_Training_310', 
                'BraTS20_Training_135', 'BraTS20_Training_129', 'BraTS20_Training_181', 'BraTS20_Training_232', 'BraTS20_Training_201', 
                'BraTS20_Training_275', 'BraTS20_Training_162', 'BraTS20_Training_358', 'BraTS20_Training_111', 'BraTS20_Training_295', 
                'BraTS20_Training_157', 'BraTS20_Training_139', 'BraTS20_Training_032', 'BraTS20_Training_040', 'BraTS20_Training_332', 
                'BraTS20_Training_351', 'BraTS20_Training_337', 'BraTS20_Training_154', 'BraTS20_Training_039', 'BraTS20_Training_312', 
                'BraTS20_Training_190', 'BraTS20_Training_170', 'BraTS20_Training_110', 'BraTS20_Training_222', 'BraTS20_Training_223', 
                'BraTS20_Training_194']
    
    brats_train_raw_dir = brats_raw_dir / "MICCAI_BraTS2020_TrainingData"

    if not brats_train_raw_dir.is_dir():
        print("WARNING: BraTS2020 cannot be setup. You first need to submit a data request form to the BraTS organizers and download the data yourself. After that is completed move the unzipped dataset to the following path and retry this script.")
        print("BraTS2020 Data Request: https://www.med.upenn.edu/cbica/brats2020/registration.html")
        print(f"BraTS2020 Destination Path: {brats_train_raw_dir}")
        return
    
    ####################################################################################################################
    #### Download BraTS2020 dataset
    ####################################################################################################################
    
    scribbles_url = "https://syncandshare.desy.de/index.php/s/DJ4KBZrZScFbTei/download/ScribbleBench_scribbles.zip"
    download(scribbles_url, archive_dir / "ScribbleBench_scribbles.zip")

    dataset_json_url = "https://syncandshare.desy.de/index.php/s/CrwGYF4EemMEXnd/download/dataset.json"
    download(dataset_json_url, brats_preprocessed_dir / "dataset.json")

    ####################################################################################################################
    #### Unpack BraTS2020 dataset archive
    ####################################################################################################################

    print("Unpacking BraTS2020 dataset archive...")

    with zipfile.ZipFile(archive_dir / "ScribbleBench_scribbles.zip", 'r') as zip_ref:
        zip_ref.extractall(raw_dir)
    
    ####################################################################################################################
    #### Preprocess BraTS2020 dataset
    ####################################################################################################################
    
    print("Preprocessing BraTS2020 dataset...")
    
    (brats_preprocessed_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (brats_preprocessed_dir / "imagesTs").mkdir(parents=True, exist_ok=True)
    (brats_preprocessed_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    (brats_preprocessed_dir / "labelsTr_dense").mkdir(parents=True, exist_ok=True)
    (brats_preprocessed_dir / "labelsTs").mkdir(parents=True, exist_ok=True)

    names = [p.name for p in brats_train_raw_dir.iterdir() if p.is_dir()]
    names = natsorted(names)

    for name in names:
        postfix = "Tr" if name not in test_set else "Ts"
        shutil.move(brats_train_raw_dir / name / f"{name}_t1.nii.gz", brats_preprocessed_dir / f"images{postfix}" / f"{name}_0000.nii.gz")
        shutil.move(brats_train_raw_dir / name / f"{name}_t1ce.nii.gz", brats_preprocessed_dir / f"images{postfix}" / f"{name}_0001.nii.gz")
        shutil.move(brats_train_raw_dir / name / f"{name}_t2.nii.gz", brats_preprocessed_dir / f"images{postfix}" / f"{name}_0002.nii.gz")
        shutil.move(brats_train_raw_dir / name / f"{name}_flair.nii.gz", brats_preprocessed_dir / f"images{postfix}" / f"{name}_0003.nii.gz")
        if postfix == "Tr":
            shutil.move(brats_train_raw_dir / name / f"{name}_seg.nii.gz", brats_preprocessed_dir / f"labels{postfix}_dense" / f"{name}.nii.gz")
        else:
            shutil.move(brats_train_raw_dir / name / f"{name}_seg.nii.gz", brats_preprocessed_dir / f"labels{postfix}" / f"{name}.nii.gz")

    names = [path.name[:-7] for path in (raw_dir / "ScribbleBench_scribbles" / "BraTS2020" / "scribblesTr").rglob("*.nii.gz")]
    for name in names:
        shutil.move(raw_dir / "ScribbleBench_scribbles" / "BraTS2020" / "scribblesTr" / f"{name}.nii.gz", brats_preprocessed_dir / "labelsTr" / f"{name}.nii.gz")

    ####################################################################################################################
    #### Delete raw dataset files
    ####################################################################################################################

    print("Deleting archive and raw dataset files...")
    shutil.rmtree(archive_dir, ignore_errors=True)
    shutil.rmtree(raw_dir, ignore_errors=True)

    print("Finished setting up BraTS2020 dataset.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset_dir", required=True, type=str, help="Path to the dir used for setting up ScribbleBench.")
    parser.add_argument('--acdc', required=False, default=False, action="store_true", help="Download and preprocess the ACDC dataset for ScribbleBench.")
    parser.add_argument('--mscmr', required=False, default=False, action="store_true", help="Download and preprocess the MSCMR dataset for ScribbleBench.")
    parser.add_argument('--word', required=False, default=False, action="store_true", help="Download and preprocess the WORD dataset for ScribbleBench.")
    parser.add_argument('--lits', required=False, default=False, action="store_true", help="Download and preprocess the LiTS dataset for ScribbleBench.")
    parser.add_argument('--brats', required=False, default=False, action="store_true", help="Download and preprocess the BraTS2020 dataset for ScribbleBench.")
    parser.add_argument('--amos', required=False, default=False, action="store_true", help="Download and preprocess the AMOS2022 dataset for ScribbleBench.")
    parser.add_argument('--kits', required=False, default=False, action="store_true", help="Download and preprocess the KiTS2023 dataset for ScribbleBench.")       
    args = parser.parse_args()

    if args.acdc:
        setup_acdc_dataset(args.dataset_dir)
    if args.mscmr:
        setup_mscmr_dataset(args.dataset_dir)
    if args.word:
        setup_word_dataset(args.dataset_dir)
    if args.lits:
        setup_lits_dataset(args.dataset_dir)
    if args.brats:
        setup_brats_dataset(args.dataset_dir)
    if args.amos:
        setup_amos_dataset(args.dataset_dir)
    if args.kits:
        setup_kits_dataset(args.dataset_dir)
