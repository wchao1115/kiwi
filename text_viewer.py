from __future__ import annotations

import sys
import tkinter as tk
from tkinter import messagebox


def show_text_window(title: str, body: str) -> None:
    if sys.platform == "win32":
        root = tk.Tk()
        root.withdraw()
        try:
            messagebox.showinfo(title, body)
        finally:
            root.destroy()
    else:
        print(title)
        print(body)
