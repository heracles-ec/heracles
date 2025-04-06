#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 08:50:13 2025

@author: ucapbba
"""
import sys
from typing import List

class logging:
    """
    Progress bar without GUI interface.
    """

    def __init__(self, label: str, *, box: List["logging"] = None) -> None:
        self.label = label
        self.current = 0
        self.total = 1  # Default to 1 to avoid division by zero
        self.box = box if box is not None else []
        self.line_offset = len(self.box)  # Track which line to overwrite
        sys.stdout.write("\n")

    def __enter__(self) -> "logging":
        # Add this instance to the box if it's not already there
        if self not in self.box:
            self.box.append(self)
        self._display_box()
        #Ensure the cursor ends at a new line after the progress bars
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._display_box()

    def update(self, current: int | None = None, total: int | None = None) -> None:
        # Update progress values
        if current is not None:
            self.current = current
        if total is not None:
            self.total = total
        # Refresh the entire display box
        self._display_box()

    def task(self, label: str) -> "logging":
        # Create a new task tied to the same box
        return self.__class__(label, box=self.box)

    def _display_box(self) -> None:
        """
        Redraw the progress bars in the terminal for all tasks in the box.
        """
        # Move the cursor up to overwrite only the progress bar lines
        sys.stdout.write(f"\033[{len(self.box)}F")  # Move up N lines
        sys.stdout.flush()
        # Display all progress bars
        for task in self.box:
            percentage = (task.current / task.total) * 100
            bar_length = 40  # Fixed number of blocks in the bar
            progress_blocks = int(percentage // (100 / bar_length))
            bar = "=" * progress_blocks + " " * (bar_length - progress_blocks)
            sys.stdout.write(f"\r{task.label}: [{bar}]\n")
        sys.stdout.flush()