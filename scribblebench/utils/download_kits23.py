"""A script to download the KiTS23 dataset into this repository"""
import sys
from tqdm import tqdm
from pathlib import Path
import urllib.request
import shutil
from time import sleep

TRAINING_CASE_NUMBERS = list(range(300)) + list(range(400, 589))


def get_destination(case_id: str, save_dir, create: bool = False):
    destination = save_dir / case_id / "imaging.nii.gz"
    if create:
        destination.parent.mkdir(exist_ok=True)
    return destination


def cleanup(tmp_pth: Path, e: Exception):
    if tmp_pth.exists():
        tmp_pth.unlink()

    if e is None:
        print("\nInterrupted.\n")
        sys.exit()
    raise(e)


def download_case(case_num: int, save_dir, pbar: tqdm, retry=True):
    remote_name = f"master_{case_num:05d}.nii.gz"
    url = f"https://kits19.sfo2.digitaloceanspaces.com/{remote_name}"
    destination = get_destination(f"case_{case_num:05d}", save_dir, True)
    tmp_pth = destination.parent / f".partial.{destination.name}"
    try:
        urllib.request.urlretrieve(url, str(tmp_pth))
        shutil.move(str(tmp_pth), str(destination))
    except KeyboardInterrupt as e:
        pbar.close()
        while True:
            try:
                sleep(0.1)
                cleanup(tmp_pth, None)
            except KeyboardInterrupt:
                pass
    except Exception as e:
        if retry:
            print(f"\nFailed to download case_{case_num:05d}. Retrying...")
            sleep(5)
            download_case(case_num, save_dir, pbar, retry=False)
        pbar.close()
        while True:
            try:
                cleanup(tmp_pth, e)
            except KeyboardInterrupt:
                pass


def download_dataset(save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Determine which cases still need to be downloaded
    left_to_download = []
    for case_num in TRAINING_CASE_NUMBERS:
        case_id = f"case_{case_num:05d}"
        dst = get_destination(case_id, save_dir)
        if not dst.exists():
            left_to_download = left_to_download + [case_num]

    # Show progressbar as cases are downloaded
    print(f"\nFound {len(left_to_download)} cases to download\n")
    for case_num in (pbar := tqdm(left_to_download)):
        pbar.set_description(f"Dowloading case_{case_num:05d}...")
        download_case(case_num, save_dir, pbar)


if __name__ == "__main__":
    download_dataset()
