"""Visual dictionary editor using tkinter."""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import Optional

from ..core.text_corrector import TextCorrector
from ..core.localization import localization


class DictionaryEditor:
    """A simple GUI for editing the custom words dictionary."""

    def __init__(self, text_corrector: TextCorrector):
        self._corrector = text_corrector
        self._window: Optional[tk.Tk] = None

    def open(self) -> None:
        """Open the editor in a new thread."""
        if self._window is not None:
            return
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def _run(self) -> None:
        """Build and run the editor window."""
        self._window = tk.Tk()
        self._window.title(localization.get("dictionary.title"))
        self._window.geometry("560x420")
        self._window.resizable(True, True)
        self._window.protocol("WM_DELETE_WINDOW", self._on_close)

        # Dark theme colors
        bg = "#2b2b2b"
        fg = "#e0e0e0"
        entry_bg = "#3c3c3c"
        select_bg = "#4a6ea8"
        btn_bg = "#3c3c3c"
        btn_active = "#505050"

        self._window.configure(bg=bg)

        style = ttk.Style(self._window)
        style.theme_use("clam")
        style.configure("Treeview",
                        background=entry_bg, foreground=fg,
                        fieldbackground=entry_bg, rowheight=26,
                        font=("Segoe UI", 10))
        style.configure("Treeview.Heading",
                        background=bg, foreground=fg,
                        font=("Segoe UI", 10, "bold"))
        style.map("Treeview",
                  background=[("selected", select_bg)],
                  foreground=[("selected", "#ffffff")])

        # --- Table ---
        table_frame = tk.Frame(self._window, bg=bg)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        self._tree = ttk.Treeview(
            table_frame,
            columns=("heard", "correction"),
            show="headings",
            selectmode="browse"
        )
        self._tree.heading("heard", text=localization.get("dictionary.col_heard"))
        self._tree.heading("correction", text=localization.get("dictionary.col_correction"))
        self._tree.column("heard", width=240, minwidth=120)
        self._tree.column("correction", width=240, minwidth=120)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=scrollbar.set)

        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        # --- Input area ---
        input_frame = tk.Frame(self._window, bg=bg)
        input_frame.pack(fill=tk.X, padx=10, pady=5)

        font = ("Segoe UI", 10)

        tk.Label(input_frame, text=localization.get("dictionary.label_heard"),
                 bg=bg, fg=fg, font=font).grid(row=0, column=0, sticky="w", padx=(0, 5))
        self._heard_var = tk.StringVar()
        self._heard_entry = tk.Entry(input_frame, textvariable=self._heard_var,
                                     bg=entry_bg, fg=fg, insertbackground=fg,
                                     font=font, relief="flat", bd=4)
        self._heard_entry.grid(row=0, column=1, sticky="ew", padx=(0, 10))

        tk.Label(input_frame, text=localization.get("dictionary.label_correction"),
                 bg=bg, fg=fg, font=font).grid(row=0, column=2, sticky="w", padx=(0, 5))
        self._correction_var = tk.StringVar()
        self._correction_entry = tk.Entry(input_frame, textvariable=self._correction_var,
                                          bg=entry_bg, fg=fg, insertbackground=fg,
                                          font=font, relief="flat", bd=4)
        self._correction_entry.grid(row=0, column=3, sticky="ew")

        input_frame.columnconfigure(1, weight=1)
        input_frame.columnconfigure(3, weight=1)

        # --- Buttons ---
        btn_frame = tk.Frame(self._window, bg=bg)
        btn_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        btn_opts = dict(bg=btn_bg, fg=fg, activebackground=btn_active,
                        activeforeground=fg, font=font, relief="flat",
                        bd=0, padx=12, pady=6, cursor="hand2")

        self._add_btn = tk.Button(btn_frame,
                                  text=localization.get("dictionary.btn_add"),
                                  command=self._add, **btn_opts)
        self._add_btn.pack(side=tk.LEFT, padx=(0, 5))

        self._delete_btn = tk.Button(btn_frame,
                                     text=localization.get("dictionary.btn_delete"),
                                     command=self._delete, **btn_opts)
        self._delete_btn.pack(side=tk.LEFT, padx=(0, 5))
        self._delete_btn.configure(state=tk.DISABLED)

        # Bind Enter key
        self._heard_entry.bind("<Return>", lambda e: self._correction_entry.focus_set())
        self._correction_entry.bind("<Return>", lambda e: self._add())

        self._populate()
        self._heard_entry.focus_set()
        self._window.mainloop()

    def _populate(self) -> None:
        """Fill the table with current corrections."""
        for item in self._tree.get_children():
            self._tree.delete(item)
        for heard, correction in sorted(self._corrector.get_corrections().items()):
            self._tree.insert("", tk.END, values=(heard, correction))

    def _on_select(self, event) -> None:
        """Handle row selection."""
        sel = self._tree.selection()
        if sel:
            values = self._tree.item(sel[0], "values")
            self._heard_var.set(values[0])
            self._correction_var.set(values[1])
            self._delete_btn.configure(state=tk.NORMAL)
        else:
            self._delete_btn.configure(state=tk.DISABLED)

    def _add(self) -> None:
        """Add or update a correction."""
        heard = self._heard_var.get().strip()
        correction = self._correction_var.get().strip()

        if not heard or not correction:
            return

        self._corrector.add_correction(heard, correction)
        self._populate()
        self._heard_var.set("")
        self._correction_var.set("")
        self._heard_entry.focus_set()
        self._delete_btn.configure(state=tk.DISABLED)

    def _delete(self) -> None:
        """Delete selected correction."""
        sel = self._tree.selection()
        if not sel:
            return
        heard = self._tree.item(sel[0], "values")[0]
        self._corrector.remove_correction(heard)
        self._populate()
        self._heard_var.set("")
        self._correction_var.set("")
        self._delete_btn.configure(state=tk.DISABLED)

    def _on_close(self) -> None:
        """Handle window close."""
        self._window.destroy()
        self._window = None
