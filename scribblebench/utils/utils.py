import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
from tqdm import tqdm


def download(
    url: str,
    output_path: Path | str,
    total_size: int | None = None,
    chunk_size: int = 8192,
    retries: int = 10,
    backoff_factor: float = 1.0,
    timeout: int | float = 60,
) -> None:
    """
    Stream-download `url` to `output_path`, resuming and retrying on failure.

    Parameters
    ----------
    url          : str
        File URL.
    output_path  : Path | str
        Target file path.
    total_size   : int | None, default None
        Known total size in bytes (overrides Content-Length).
    chunk_size   : int, default 8192
        Bytes per read.
    retries      : int, default 10
        Max retry attempts.
    backoff_factor : float, default 1.0
        Exponential back-off between retries.
    timeout      : int | float, default 60
        Seconds for connect/read timeout.
    """
    output_path = Path(output_path)
    headers = {}

    # -- Resume if partial file exists ---------------------------------------
    existing = output_path.stat().st_size if output_path.exists() else 0
    if existing:
        headers["Range"] = f"bytes={existing}-"

    # -- Configure session w/ retries ----------------------------------------
    sess = requests.Session()
    retry_cfg = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    sess.mount("https://", HTTPAdapter(max_retries=retry_cfg))
    sess.mount("http://",  HTTPAdapter(max_retries=retry_cfg))

    # -- Request -------------------------------------------------------------
    with sess.get(url, headers=headers, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()

        # Determine final expected size
        if total_size is None:
            content_len = resp.headers.get("Content-Length")
            total = (int(content_len) + existing) if content_len else None
        else:
            total = total_size

        # -- Download w/ progress bar ---------------------------------------
        mode = "ab" if existing else "wb"
        with open(output_path, mode) as fh, tqdm(
            total=total,
            initial=existing,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=output_path.name,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:                    # filter out keep-alive chunks
                    fh.write(chunk)
                    bar.update(len(chunk))


def info2dict(filepath):
    with open((filepath.parent / "Info.cfg"), "r") as f:
        info = f.read()

    data = {}
    for line in info.strip().splitlines():
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        # Convert value to int or float if possible
        if value.replace('.', '', 1).isdigit():
            value = float(value) if '.' in value else int(value)

        data[key] = value
    return data


# ---------------- example usage ----------------
if __name__ == "__main__":
    URL  = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/collection/637218c173e9f0047faa00fb/download"
    FILE = Path("archives/ACDC.zip")
    FILE.parent.mkdir(exist_ok=True)

    # If you know the size (~1.6 GB) pass it:
    download(URL, FILE, total_size=int(1.6 * 1024**3))
