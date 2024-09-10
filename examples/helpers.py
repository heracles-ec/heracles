"""
Helper functions for examples.
"""

import os
import urllib.request
import hashlib
import heracles


def checksum(
    path: str,
    *,
    md5: str | None = None,
) -> bool:
    """
    Check a local file against a given checksum.
    """

    md5sum = hashlib.md5() if md5 else None

    with open(path, "rb") as fp:
        while block := fp.read(4096):
            if md5sum:
                md5sum.update(block)

    if md5sum and md5sum.hexdigest() != md5:
        return False

    return True


def download(
    path: str,
    url: str,
    *,
    overwrite: bool = False,
    keep: bool = False,
    md5: str | None = None,
    progress: heracles.Progress | None = None,
) -> None:
    """
    Download a file.
    """

    def reporthook(count: int, block_size: int, total_size: int) -> None:
        if progress is not None:
            current = count * block_size
            if total_size > -1 and current > total_size:
                current = total_size
            progress.update(current, total_size if total_size > -1 else None)

    if os.path.exists(path):
        if checksum(path, md5=md5):
            return
        if not overwrite:
            raise FileExistsError(path)
    good = False
    try:
        urllib.request.urlretrieve(url, path, reporthook=reporthook)
        if not checksum(path, md5=md5):
            raise ValueError(f"{path}: invalid checksum, removing")
        good = True
    finally:
        if not good and not keep:
            os.unlink(path)


def get_example_data(progress: heracles.Progress | None = None) -> None:
    """
    Download the example catalogue and associated products.
    """
    download(
        "catalog.fits",
        "https://zenodo.org/records/13622599/files/catalog.fits?download=1&preview=1",
        md5="e6c469425a6c072b4736aa6e5511bc28",
        progress=progress,
    )
    download(
        "vmap.fits.gz",
        "https://zenodo.org/records/13622599/files/footprint.fits.gz?download=1&preview=1",
        md5="f6d201bdf57a8e8a9a800ab1e7f1095f",
        progress=progress,
    )
    download(
        "nz.npz",
        "https://zenodo.org/records/13622599/files/nz.npz?download=1&preview=1",
        md5="57bbbce4e21ce34ddf460dac05363b37",
        progress=progress,
    )
