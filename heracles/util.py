# Heracles: Euclid code for harmonic-space statistics on the sphere
#
# Copyright (C) 2023 Euclid Science Ground Segment
#
# This file is part of Heracles.
#
# Heracles is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Heracles is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Heracles. If not, see <https://www.gnu.org/licenses/>.
"""module for utilities"""

import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import timedelta


def toc_match(key, include=None, exclude=None):
    """return whether a tocdict entry matches include/exclude criteria"""
    if include is not None:
        for pattern in include:
            if all(p is Ellipsis or p == k for p, k in zip(pattern, key)):
                break
        else:
            return False
    if exclude is not None:
        for pattern in exclude:
            if all(p is Ellipsis or p == k for p, k in zip(pattern, key)):
                return False
    return True


def toc_filter(obj, include=None, exclude=None):
    """return a filtered toc dict ``d``"""
    if isinstance(obj, Sequence):
        return [toc_filter(item, include, exclude) for item in obj]
    if isinstance(obj, Mapping):
        return {k: v for k, v in obj.items() if toc_match(k, include, exclude)}
    msg = "invalid input type"
    raise TypeError(msg)


class tocdict(dict):
    """ToC dictionary with filtering"""

    def __getitem__(self, pattern):
        try:
            return super().__getitem__(pattern)
        except (KeyError, TypeError):
            pass
        if not isinstance(pattern, tuple):
            pattern = (pattern,)
        found = self.__class__()
        for key, value in self.items():
            if isinstance(key, tuple):
                if len(key) < len(pattern):
                    continue
                if all(p == k for p, k in zip(pattern, key) if p is not ...):
                    found[key] = value
            else:
                if pattern == (...,) or pattern == (key,):
                    found[key] = value
        if not found:
            raise KeyError(pattern)
        return found

    def __or__(self, other):
        result = tocdict(self)
        result.update(other)
        return result


class Progress:
    """simple progress bar for operations"""

    def __init__(self, out=sys.stdout):
        """create a new progress bar"""
        self.out = out
        self.time = 0
        self.progress = 0
        self.total = 0
        self.title = None

    def start(self, total, title=None):
        """start new progress"""
        self.time = time.monotonic()
        self.progress = 0
        self.total = total
        self.title = title
        self.update(0)

    def update(self, step=1):
        """update progress"""
        self.progress = min(self.progress + step, self.total)
        m = f"{self.title!s}: " if self.title is not None else ""
        p = self.progress / self.total
        b = "#" * int(20 * p)
        f = f"{self.progress:_}/{self.total:_}"
        t = timedelta(seconds=(time.monotonic() - self.time))
        s = f"\r{m}{100*p:3.0f}% |{b:20s}| {f} | {t}"
        try:
            w, _ = os.get_terminal_size(self.out.fileno())
        except (OSError, AttributeError):
            pass
        else:
            if w > 0:
                s = s[:w]
        self.out.write(s)
        self.out.flush()

    def stop(self, complete=True):
        """stop progress and end line"""
        if complete:
            self.update(self.total - self.progress)
        self.out.write("\n")
