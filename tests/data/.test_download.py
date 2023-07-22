import glob
import os

from src.data.download import download_file, download_altlex, download_ctb, download_because2

THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR: str = "../materials/"


def test_download_file() -> None:
    zip_url: str = "https://www.applican.com/download/zip/sample/web.zip"
    save_path: str = "../materials/sample-zip-file.zip"
    download_file(zip_url, save_path)
    assert os.path.isfile(save_path)


def test_download_altlex() -> None:
    download_altlex(DOWNLOAD_DIR)
    for fname in ("altlex_dev", "altlex_gold"):
        for ext in (".tsv", ".tsv.zip"):
            assert os.path.isfile(os.path.join(DOWNLOAD_DIR, fname + ext))


def test_download_ctb() -> None:
    subdir: str = "ctb"
    download_ctb(DOWNLOAD_DIR, subdir)
    lst_tml: list[str] = sorted(glob.glob(os.path.join(DOWNLOAD_DIR, subdir, "*.tml")))
    assert len(lst_tml) == 183
    assert os.path.basename(lst_tml[0]) == "ABC19980108.1830.0711.tml"


def test_download_because2() -> None:
    download_because2(DOWNLOAD_DIR)
    assert os.path.isdir(os.path.join(DOWNLOAD_DIR, "BECAUSE-2.0"))

