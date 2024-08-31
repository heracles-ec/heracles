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
