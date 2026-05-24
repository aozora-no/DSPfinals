import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import cv2
import soundfile as sf
from scipy.signal import butter, lfilter, cheby1, bessel
import os
from PIL import Image, ImageTk
import io
import re
import time

# Set Matplotlib to a cleaner style for a professional look
plt.style.use('bmh')


class DSPMasterProgram:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Digital Signal Processing Suite")
        self.root.geometry("1280x850")
        self.root.configure(bg="#1e272e") # Deep Dark background

        self.apply_custom_styles()

        # Header Area
        header = tk.Frame(root, bg="#2f3640", height=60)
        header.pack(fill='x', side='top')
        tk.Label(header, text="CORE DSP MASTER ENGINE", font=("Segoe UI", 18, "bold"),
                 fg="#00d8d6", bg="#2f3640").pack(pady=10, padx=20, side='left')
        self.back_button = tk.Button(header, text="← Back to Labs", command=self.show_front_page,
                                     bg="#00d8d6", fg="#1e272e", activebackground="#34e7e4",
                                     activeforeground="#1e272e", relief="flat",
                                     font=("Segoe UI", 10, "bold"), padx=14, pady=6,
                                     cursor="hand2")

        # Front page and hidden-tab lab container
        self.home_frame = tk.Frame(root, bg="#1e272e")
        self.notebook = ttk.Notebook(root, style="Hidden.TNotebook")

        # Tabs initialization
        self.create_lab1_tab()
        self.create_image_processing_tab()
        self.create_audio_processing_tab()
        self.create_z_transform_tab()
        self.create_inverse_z_transform_tab()
        self.create_dft_lab5_tab()
        self.create_fft_tab()
        self.create_windowing_tab()
        self.create_fir_iir_lab8_tab()
        self.create_front_page()
        self.show_front_page()
        self.root.bind("<Escape>", lambda event: self.show_front_page())

    def apply_custom_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        # Notebook Styling
        style.configure("TNotebook", background="#1e272e", borderwidth=0)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[10, 5], background="#485460", foreground="white")
        style.map("TNotebook.Tab", background=[("selected", "#00d8d6")], foreground=[("selected", "#1e272e")])
        style.configure("Hidden.TNotebook", background="#1e272e", borderwidth=0)
        style.layout("Hidden.TNotebook.Tab", [])

        # Frame Styling
        style.configure("TFrame", background="#f1f2f6")
        style.configure("Card.TFrame", background="white", relief="flat", borderwidth=1)

        # LabelFrame Styling (The Dashboard Cards)
        style.configure("TLabelframe", background="#f1f2f6", relief="groove")
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"), foreground="#2f3542", background="#f1f2f6")

        # Button Styling
        style.configure("TButton", font=("Segoe UI", 9, "bold"), padding=5)
        style.configure("Action.TButton", background="#3498db", foreground="white")

    def create_front_page(self):
        hero = tk.Frame(self.home_frame, bg="#1e272e")
        hero.pack(fill='x', padx=34, pady=(30, 14))

        tk.Label(hero, text="Choose a DSP Laboratory", font=("Segoe UI", 30, "bold"),
                 fg="white", bg="#1e272e").pack(anchor='w')
        tk.Label(hero, text="Open one module at a time. Use the Back button to return to this lab launcher.",
                 font=("Segoe UI", 12), fg="#d2dae2", bg="#1e272e").pack(anchor='w', pady=(6, 0))

        lab_grid = tk.Frame(self.home_frame, bg="#1e272e")
        lab_grid.pack(fill='both', expand=True, padx=28, pady=(4, 30))

        labs = [
            ("📈", "Lab 1", "Signal Plotter", "Continuous, sampled, and overlay waveform synthesis.", "#00d8d6"),
            ("🖼️", "Lab 2", "Imaging", "Load images and apply grayscale, binary, and filter tools.", "#54a0ff"),
            ("🔊", "Lab 3", "Audio", "Import WAV files and inspect lowpass filtering results.", "#1dd1a1"),
            ("Ω", "Lab 4", "Z-Transform", "Generate symbolic Z expressions from discrete sequences.", "#feca57"),
            ("Ω", "Lab 4B", "Inverse Z", "Extract sequence values from Z-domain expressions.", "#ff9f43"),
            ("⚙️", "Lab 5", "DFT Analysis", "Run manual DFT checks and vibration spectrum simulations.", "#ff6b6b"),
            ("⚡", "Lab 6", "FFT vs DFT", "Compare direct DFT loops against fast FFT computation.", "#5f27cd"),
            ("🪟", "Lab 7", "Windowing", "Study window functions and spectral leakage behavior.", "#48dbfb"),
            ("⚖️", "Lab 8", "FIR vs IIR", "Compare filter response and design characteristics.", "#c8d6e5"),
        ]

        for col in range(3):
            lab_grid.grid_columnconfigure(col, weight=1, uniform="labcards")
        for row in range(3):
            lab_grid.grid_rowconfigure(row, weight=1, uniform="labcards")

        for index, (icon, lab, title, desc, accent) in enumerate(labs):
            row = index // 3
            col = index % 3
            card = tk.Frame(lab_grid, bg="#2f3640", highlightthickness=1,
                            highlightbackground="#485460", cursor="hand2")
            card.grid(row=row, column=col, sticky="nsew", padx=10, pady=10)
            card.grid_propagate(False)

            accent_bar = tk.Frame(card, bg=accent, height=4)
            accent_bar.pack(fill='x')

            body = tk.Frame(card, bg="#2f3640")
            body.pack(fill='both', expand=True, padx=18, pady=16)

            tk.Label(body, text=icon, font=("Segoe UI", 24), fg=accent, bg="#2f3640").pack(anchor='w')
            tk.Label(body, text=lab, font=("Segoe UI", 10, "bold"), fg=accent, bg="#2f3640").pack(anchor='w', pady=(8, 0))
            tk.Label(body, text=title, font=("Segoe UI", 16, "bold"), fg="white", bg="#2f3640").pack(anchor='w')
            tk.Label(body, text=desc, font=("Segoe UI", 10), fg="#d2dae2", bg="#2f3640",
                     wraplength=310, justify='left').pack(anchor='w', pady=(8, 0))

            for widget in (card, accent_bar, body, *body.winfo_children()):
                widget.bind("<Button-1>", lambda event, i=index: self.open_lab(i))
                widget.bind("<Enter>", lambda event, c=card, b=body, a=accent: self._set_card_hover(c, b, a, True))
                widget.bind("<Leave>", lambda event, c=card, b=body, a=accent: self._set_card_hover(c, b, a, False))

    def _set_card_hover(self, card, body, accent, hover):
        color = "#3d4854" if hover else "#2f3640"
        card.configure(bg=color, highlightbackground=accent if hover else "#485460")
        body.configure(bg=color)
        for child in body.winfo_children():
            child.configure(bg=color)

    def open_lab(self, index):
        self.home_frame.pack_forget()
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        self.notebook.select(index)
        self.back_button.pack(pady=10, padx=20, side='right')

    def show_front_page(self):
        self.notebook.pack_forget()
        self.back_button.pack_forget()
        self.home_frame.pack(fill='both', expand=True)

    # ==================== TAB 0: LAB 1 - CUSTOM SIGNAL PLOTTER ====================
    def create_lab1_tab(self):
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="📈 Lab 1: Signal Plotter")

        # ---- Parameters frame ----
        param_frame = ttk.LabelFrame(frame, text="Signal Parameters", padding="10")
        param_frame.pack(fill='x', pady=(0, 5))

        # Row 1: Signal type, frequency, duration
        row1 = ttk.Frame(param_frame)
        row1.pack(fill='x', pady=3)

        ttk.Label(row1, text="Signal Type:").pack(side='left', padx=(0, 4))
        self.lab1_signal_type = ttk.Combobox(row1, width=10, state='readonly',
            values=["Cosine", "Sine"])
        self.lab1_signal_type.current(0)
        self.lab1_signal_type.pack(side='left', padx=(0, 15))

        ttk.Label(row1, text="Frequency (Hz):").pack(side='left', padx=(0, 4))
        self.lab1_freq = ttk.Entry(row1, width=8)
        self.lab1_freq.insert(0, "5")
        self.lab1_freq.pack(side='left', padx=(0, 15))

        ttk.Label(row1, text="Duration (s):").pack(side='left', padx=(0, 4))
        self.lab1_duration = ttk.Entry(row1, width=8)
        self.lab1_duration.insert(0, "1")
        self.lab1_duration.pack(side='left', padx=(0, 15))

        ttk.Label(row1, text="Amplitude:").pack(side='left', padx=(0, 4))
        self.lab1_amplitude = ttk.Entry(row1, width=8)
        self.lab1_amplitude.insert(0, "1")
        self.lab1_amplitude.pack(side='left')

        # Row 2: Sample rate (only used for sampled plots)
        row2 = ttk.Frame(param_frame)
        row2.pack(fill='x', pady=3)

        ttk.Label(row2, text="Sample Rate fs (Hz):").pack(side='left', padx=(0, 4))
        self.lab1_fs = ttk.Entry(row2, width=8)
        self.lab1_fs.insert(0, "8000")
        self.lab1_fs.pack(side='left', padx=(0, 15))

        ttk.Label(row2, text="(used for Sampled and Both plots)",
                  foreground="gray").pack(side='left')

        # ---- Plot type selection ----
        plot_frame = ttk.LabelFrame(frame, text="Plot Type", padding="10")
        plot_frame.pack(fill='x', pady=(0, 5))

        btn_row = ttk.Frame(plot_frame)
        btn_row.pack(fill='x')

        ttk.Button(btn_row, text="📈 Continuous Only",
                   command=lambda: self.lab1_plot("continuous")).pack(side='left', padx=5)
        ttk.Button(btn_row, text="📊 Sampled Only",
                   command=lambda: self.lab1_plot("sampled")).pack(side='left', padx=5)
        ttk.Button(btn_row, text="📈📊 Both (Continuous + Sampled)",
                   command=lambda: self.lab1_plot("both")).pack(side='left', padx=5)
        tk.Button(btn_row, text="🗑️ Clear", command=self.lab1_delete,
                  fg="red", bg="#f0f0f0", relief='raised', cursor="hand2").pack(side='left', padx=15)

        # ---- Examples section ----
        # Preset: (label, signal_type, freq, duration, amplitude, fs, plot_mode)
        self.lab1_presets = [
            ("Simple 5Hz Cosine",         "Cosine", 5,    1,    1, 1000, "continuous"),
            ("Simple 10Hz Sine",          "Sine",   10,   1,    2, 1000, "continuous"),
            ("1kHz @ 8kHz (Nyquist OK)",  "Cosine", 1000, 0.01, 1, 8000, "both"),
            ("2kHz @ 8kHz (Nyquist OK)",  "Cosine", 2000, 0.01, 1, 8000, "both"),
            ("6kHz @ 8kHz (Aliasing ⚠️)", "Cosine", 6000, 0.01, 1, 8000, "both"),
            ("5kHz @ 8kHz (Aliasing ⚠️)", "Cosine", 5000, 0.01, 1, 8000, "both"),
        ]

        ex_frame = ttk.LabelFrame(frame, text="Examples  (auto-fills parameters above)", padding="8")
        ex_frame.pack(fill='x', pady=(0, 5))

        ex_row = ttk.Frame(ex_frame)
        ex_row.pack(fill='x')

        for preset in self.lab1_presets:
            lbl = preset[0]
            ttk.Button(ex_row, text=lbl,
                       command=lambda p=preset: self.lab1_load_preset(p)
                       ).pack(side='left', padx=4)

        # ---- Status / info label ----
        self.lab1_desc = ttk.Label(frame, text="Set parameters above and choose a plot type.",
                                   anchor='w', foreground="gray")
        self.lab1_desc.pack(fill='x', padx=5, pady=(0, 3))

        # ---- Matplotlib canvas ----
        self.lab1_fig = Figure(figsize=(10, 5), dpi=80)
        self.lab1_canvas = FigureCanvasTkAgg(self.lab1_fig, master=frame)
        self.lab1_canvas.get_tk_widget().pack(fill='both', expand=True)

    def lab1_delete(self):
        self.lab1_fig.clear()
        self.lab1_canvas.draw()
        self.lab1_desc.config(text="Set parameters above and choose a plot type.", foreground="gray")

    def lab1_load_preset(self, preset):
        """Fill all parameter fields from preset tuple and auto-plot."""
        _, sig_type, freq, duration, amplitude, fs, plot_mode = preset
        self.lab1_signal_type.set(sig_type)
        for entry, val in [
            (self.lab1_freq,      freq),
            (self.lab1_duration,  duration),
            (self.lab1_amplitude, amplitude),
            (self.lab1_fs,        fs),
        ]:
            entry.delete(0, tk.END)
            entry.insert(0, str(val))
        self.lab1_plot(plot_mode)

    def lab1_get_params(self):
        """Read and validate all Lab 1 input fields. Returns dict or None on error."""
        try:
            f    = float(self.lab1_freq.get())
            dur  = float(self.lab1_duration.get())
            amp  = float(self.lab1_amplitude.get())
            fs   = float(self.lab1_fs.get())
            sig  = self.lab1_signal_type.get()
            if f <= 0 or dur <= 0 or amp <= 0 or fs <= 0:
                raise ValueError("All values must be positive.")
            return {"f": f, "dur": dur, "amp": amp, "fs": fs, "sig": sig}
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please check your parameters.\n{e}")
            return None

    def lab1_make_signal(self, params, t):
        fn = np.cos if params["sig"] == "Cosine" else np.sin
        return params["amp"] * fn(2 * np.pi * params["f"] * t)

    def lab1_plot(self, mode):
        p = self.lab1_get_params()
        if p is None:
            return

        f, dur, amp, fs, sig = p["f"], p["dur"], p["amp"], p["fs"], p["sig"]

        # Continuous time axis
        t_cont = np.linspace(0, dur, max(1000, int(dur * 5000)))
        x_cont = self.lab1_make_signal(p, t_cont)

        # Sampled time axis
        t_samp = np.arange(0, dur, 1 / fs)
        x_samp = self.lab1_make_signal(p, t_samp)

        # Nyquist check
        nyquist_ok = fs >= 2 * f
        nyquist_msg = (f"✅ Nyquist satisfied  (fs={fs:.0f} ≥ 2f={2*f:.0f})"
                       if nyquist_ok else
                       f"⚠️  ALIASING!  Nyquist violated  (fs={fs:.0f} < 2f={2*f:.0f})")
        alias_f = abs(f - fs) if not nyquist_ok else None

        self.lab1_fig.clear()

        if mode == "continuous":
            ax = self.lab1_fig.add_subplot(111)
            ax.plot(t_cont, x_cont, 'b-', linewidth=2)
            ax.set_title(f"{sig} Signal  —  f = {f} Hz,  A = {amp},  Duration = {dur} s")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            info = f"Continuous {sig.lower()} wave | f={f} Hz | A={amp} | duration={dur}s"

        elif mode == "sampled":
            ax = self.lab1_fig.add_subplot(111)
            ax.stem(t_samp, x_samp, linefmt='r-', markerfmt='go', basefmt='b')
            title = f"Sampled {sig}  —  f={f} Hz  @  fs={fs:.0f} Hz"
            if not nyquist_ok:
                title = "⚠️  " + title + "  (ALIASING)"
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            info = nyquist_msg
            if alias_f:
                info += f"  |  Alias freq ≈ {alias_f:.0f} Hz"

        else:  # both
            ax = self.lab1_fig.add_subplot(111)
            ax.plot(t_cont, x_cont, 'b-', linewidth=2, label="Continuous")
            ax.stem(t_samp, x_samp, linefmt='r-', markerfmt='go', basefmt='b',
                    label=f"Sampled (fs={fs:.0f} Hz)")
            title = f"{sig}  f={f} Hz  |  fs={fs:.0f} Hz"
            if not nyquist_ok:
                title = "⚠️  " + title + "  — ALIASING"
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.legend()
            ax.grid(True, alpha=0.3)
            info = nyquist_msg
            if alias_f:
                info += f"  |  Alias freq ≈ {alias_f:.0f} Hz"

        self.lab1_fig.tight_layout()
        self.lab1_canvas.draw()
        color = "green" if (mode == "continuous" or nyquist_ok) else "red"
        self.lab1_desc.config(text=info, foreground=color)

    # ==================== TAB 1: IMAGE PROCESSING ====================
    def create_image_processing_tab(self):
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="🖼️ Lab 2: Imaging")

        # ── Top: filter buttons ──────────────────────────────────────────
        control_frame = ttk.LabelFrame(frame, text="Image Processing Controls", padding="10")
        control_frame.pack(fill='x', pady=(0, 4))

        ttk.Button(control_frame, text="Load Image",
                   command=self.load_image).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Grayscale",
                   command=lambda: self.select_filter("Grayscale")).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Binary (B&W)",
                   command=lambda: self.select_filter("Binary")).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Median Blur",
                   command=lambda: self.select_filter("MedianBlur")).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Laplacian",
                   command=lambda: self.select_filter("Laplacian")).pack(side='left', padx=5)
        tk.Button(control_frame, text="🗑️ Delete", command=self.delete_image,
                  fg="red", bg="#f0f0f0", relief='raised', cursor="hand2").pack(side='left', padx=5)

        # ── Dynamic filter options panel (swaps contents per filter) ─────
        self.filter_options_frame = ttk.LabelFrame(
            frame, text="Filter Options  —  select a filter above", padding="8")
        self.filter_options_frame.pack(fill='x', pady=(0, 4))

        # Placeholder label shown when no filter is selected
        self.filter_placeholder = ttk.Label(
            self.filter_options_frame,
            text="No filter selected. Click a filter button to see its options.",
            foreground="gray")
        self.filter_placeholder.pack(anchor='w')

        # ── Split-screen centred ─────────────────────────────────────────
        outer = ttk.Frame(frame)
        outer.pack(fill='both', expand=True)

        self.img_split_frame = ttk.Frame(outer)
        self.img_split_frame.place(relx=0.5, rely=0.5, anchor='center')

        # Left — Original
        left_panel = ttk.Frame(self.img_split_frame, relief='groove', borderwidth=2)
        left_panel.pack(side='left', padx=10, pady=10)
        ttk.Label(left_panel, text="Original Image",
                  font=("Helvetica", 10, "bold")).pack(pady=(6, 2))
        self.img_original_label = ttk.Label(
            left_panel, text="Load an image to begin",
            background="#dcdcdc", width=42, anchor='center')
        self.img_original_label.pack(padx=8, pady=(0, 8), ipadx=4, ipady=4)

        # Divider
        ttk.Separator(self.img_split_frame, orient='vertical').pack(
            side='left', fill='y', pady=10)

        # Right — Processed
        right_panel = ttk.Frame(self.img_split_frame, relief='groove', borderwidth=2)
        right_panel.pack(side='left', padx=10, pady=10)
        ttk.Label(right_panel, text="Processed Image",
                  font=("Helvetica", 10, "bold")).pack(pady=(6, 2))
        self.img_processed_label = ttk.Label(
            right_panel, text="Apply a filter to see result",
            background="#dcdcdc", width=42, anchor='center')
        self.img_processed_label.pack(padx=8, pady=(0, 8), ipadx=4, ipady=4)

        # Status bar
        self.img_status = ttk.Label(frame, text="", anchor='center', foreground="gray")
        self.img_status.pack(fill='x', pady=(0, 4))

        self.original_image   = None
        self.current_image    = None
        self.current_filter   = None

        # ── Per-filter tk variables (created once, reused) ───────────────
        # Grayscale
        self.gs_method = tk.StringVar(value="Weighted (Standard)")

        # Binary
        self.bin_thresh    = tk.IntVar(value=127)
        self.bin_type      = tk.StringVar(value="Binary")

        # Median Blur
        self.blur_ksize    = tk.IntVar(value=9)

        # Laplacian
        self.lap_ksize     = tk.IntVar(value=1)
        self.lap_scale     = tk.DoubleVar(value=1.0)
        self.lap_delta     = tk.DoubleVar(value=0.0)

    # ── Dynamic options panel builder ────────────────────────────────────
    def select_filter(self, filter_name):
        """Show the correct options panel for filter_name and apply the filter."""
        self.current_filter = filter_name

        # Destroy all current children of the options frame
        for w in self.filter_options_frame.winfo_children():
            w.destroy()

        self.filter_options_frame.config(text=f"Filter Options  —  {filter_name}")

        if filter_name == "Grayscale":
            self._build_grayscale_options()
        elif filter_name == "Binary":
            self._build_binary_options()
        elif filter_name == "MedianBlur":
            self._build_blur_options()
        elif filter_name == "Laplacian":
            self._build_laplacian_options()

        # Apply immediately with current settings
        self.apply_current_filter()

    def _apply_btn(self, parent):
        """Reusable Apply button."""
        ttk.Button(parent, text="▶ Apply",
                   command=self.apply_current_filter).pack(side='left', padx=10)

    def _build_grayscale_options(self):
        f = self.filter_options_frame
        ttk.Label(f, text="Method:").pack(side='left', padx=(0, 4))
        cb = ttk.Combobox(f, textvariable=self.gs_method, width=20, state='readonly',
                          values=["Weighted (Standard)", "Average", "Luminosity"])
        cb.pack(side='left', padx=(0, 10))
        self._apply_btn(f)
        ttk.Label(f, text="Weighted=standard OpenCV  |  Average=(R+G+B)/3  |  Luminosity=perceived brightness",
                  foreground="gray").pack(side='left', padx=10)

    def _build_binary_options(self):
        f = self.filter_options_frame

        ttk.Label(f, text="Threshold (0–255):").pack(side='left', padx=(0, 4))
        ttk.Spinbox(f, from_=0, to=255, textvariable=self.bin_thresh,
                    width=5).pack(side='left', padx=(0, 10))

        ttk.Label(f, text="Type:").pack(side='left', padx=(0, 4))
        ttk.Combobox(f, textvariable=self.bin_type, width=18, state='readonly',
                     values=["Binary", "Binary Inverted", "Truncate", "To-Zero"]
                     ).pack(side='left', padx=(0, 10))

        self._apply_btn(f)
        ttk.Label(f, text="Lower threshold = more white pixels",
                  foreground="gray").pack(side='left', padx=10)

    def _build_blur_options(self):
        f = self.filter_options_frame

        ttk.Label(f, text="Kernel Size (odd):").pack(side='left', padx=(0, 4))
        ttk.Combobox(f, textvariable=self.blur_ksize, width=6, state='readonly',
                     values=[3, 5, 7, 9, 11, 13, 15]
                     ).pack(side='left', padx=(0, 10))

        self._apply_btn(f)
        ttk.Label(f, text="Larger kernel = stronger blur / more noise removal",
                  foreground="gray").pack(side='left', padx=10)

    def _build_laplacian_options(self):
        f = self.filter_options_frame

        ttk.Label(f, text="ksize:").pack(side='left', padx=(0, 4))
        ttk.Combobox(f, textvariable=self.lap_ksize, width=4, state='readonly',
                     values=[1, 3, 5, 7]).pack(side='left', padx=(0, 10))

        ttk.Label(f, text="Scale:").pack(side='left', padx=(0, 4))
        ttk.Spinbox(f, from_=0.1, to=10.0, increment=0.1,
                    textvariable=self.lap_scale, width=5, format="%.1f"
                    ).pack(side='left', padx=(0, 10))

        ttk.Label(f, text="Delta:").pack(side='left', padx=(0, 4))
        ttk.Spinbox(f, from_=-128, to=128, increment=1,
                    textvariable=self.lap_delta, width=5
                    ).pack(side='left', padx=(0, 10))

        self._apply_btn(f)
        ttk.Label(f, text="ksize=1 → sharp edges  |  larger ksize → broader edge detection",
                  foreground="gray").pack(side='left', padx=10)

    # ── Filter application ───────────────────────────────────────────────
    def apply_current_filter(self):
        f = self.current_filter
        if f == "Grayscale":
            self.apply_grayscale()
        elif f == "Binary":
            self.apply_binary()
        elif f == "MedianBlur":
            self.apply_median_blur()
        elif f == "Laplacian":
            self.apply_laplacian()

    # ── Image helpers ────────────────────────────────────────────────────
    def _to_photoimage(self, bgr, size=(380, 290)):
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        return ImageTk.PhotoImage(Image.fromarray(img))

    def delete_image(self):
        self.original_image = None
        self.current_image  = None
        self.img_original_label.config(image='', text="Load an image to begin")
        self.img_processed_label.config(image='', text="Apply a filter to see result")
        self.img_status.config(text="")
        # Reset options panel
        for w in self.filter_options_frame.winfo_children():
            w.destroy()
        self.filter_options_frame.config(text="Filter Options  —  select a filter above")
        ttk.Label(self.filter_options_frame,
                  text="No filter selected. Click a filter button to see its options.",
                  foreground="gray").pack(anchor='w')
        self.current_filter = None
        messagebox.showinfo("Success", "Image data cleared!")

    def load_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.bmp"), ("All files", "*.*")])
        if not filepath:
            return
        self.original_image = cv2.imread(filepath)
        self.current_image  = self.original_image.copy()
        photo = self._to_photoimage(self.original_image)
        self.img_original_label.config(image=photo, text="")
        self.img_original_label.image = photo
        self.img_processed_label.config(image='', text="Apply a filter to see result")
        self.img_status.config(
            text=f"Loaded: {os.path.basename(filepath)}", foreground="gray")

    def display_processed(self, bgr, label_text):
        photo = self._to_photoimage(bgr)
        self.img_processed_label.config(image=photo, text="")
        self.img_processed_label.image = photo
        self.img_status.config(text=f"Filter applied: {label_text}", foreground="blue")

    def apply_grayscale(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first"); return
        try:
            method = self.gs_method.get()
            b, g, r = cv2.split(self.original_image.astype(np.float32))
            if method == "Average":
                gray = np.uint8((b + g + r) / 3)
            elif method == "Luminosity":
                gray = np.uint8(0.0722*b + 0.7152*g + 0.2126*r)
            else:  # Weighted (Standard)
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.current_image = gray
            self.display_processed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                                   f"Grayscale ({method})")
        except (ValueError, TypeError) as e:
            messagebox.showerror("Type Error", f"Grayscale parameter error:\n{e}")

    def apply_binary(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first"); return
        try:
            thresh = int(self.bin_thresh.get())
            if not (0 <= thresh <= 255):
                raise ValueError("Threshold must be between 0 and 255.")
            type_map = {
                "Binary":          cv2.THRESH_BINARY,
                "Binary Inverted": cv2.THRESH_BINARY_INV,
                "Truncate":        cv2.THRESH_TRUNC,
                "To-Zero":         cv2.THRESH_TOZERO,
            }
            t = type_map.get(self.bin_type.get(), cv2.THRESH_BINARY)
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, thresh, 255, t)
            self.current_image = bw
            self.display_processed(cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR),
                                   f"Binary ({self.bin_type.get()}, thresh={thresh})")
        except (ValueError, TypeError) as e:
            messagebox.showerror("Type Error", f"Binary parameter error:\n{e}")

    def apply_median_blur(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first"); return
        try:
            k = int(self.blur_ksize.get())
            if k % 2 == 0 or k < 1:
                raise ValueError("Kernel size must be a positive odd number (3, 5, 7 ...).")
            gray    = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.medianBlur(gray, k)
            self.current_image = blurred
            self.display_processed(cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR),
                                   f"Median Blur (ksize={k})")
        except (ValueError, TypeError) as e:
            messagebox.showerror("Type Error", f"Median Blur parameter error:\n{e}")

    def apply_laplacian(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first"); return
        try:
            k     = int(self.lap_ksize.get())
            scale = float(self.lap_scale.get())
            delta = float(self.lap_delta.get())
            if k % 2 == 0 or k < 1:
                raise ValueError("ksize must be 1, 3, 5, or 7.")
            if scale <= 0:
                raise ValueError("Scale must be a positive number.")
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            lap  = cv2.Laplacian(gray, cv2.CV_64F, ksize=k, scale=scale, delta=delta)
            lap8 = np.uint8(np.clip(np.absolute(lap), 0, 255))
            self.current_image = lap8
            self.display_processed(cv2.cvtColor(lap8, cv2.COLOR_GRAY2BGR),
                                   f"Laplacian (ksize={k}, scale={scale}, delta={delta})")
        except (ValueError, TypeError) as e:
            messagebox.showerror("Type Error", f"Laplacian parameter error:\n{e}")

    # ==================== TAB 2: AUDIO PROCESSING (ENHANCED) ====================
    def create_audio_processing_tab(self):
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="🔊 Lab 3: Audio")

        # ── Internal state (MUST come first) ──────────────────────────────
        self.audio_data = None
        self.audio_fs = None
        self.filtered_audio = None

        # ── Filter parameters (tk variables) ─────────────────────────────
        self.audio_cutoff = tk.DoubleVar(value=1000.0)
        self.audio_cutoff_low = tk.DoubleVar(value=1000.0)
        self.audio_cutoff_high = tk.DoubleVar(value=3000.0)
        self.audio_filter_order = tk.IntVar(value=5)
        self.audio_design_method = tk.StringVar(value="Butterworth")

        # ── Top: File Loading ────────────────────────────────────────────
        control_frame = ttk.LabelFrame(frame, text="Audio Processing Controls", padding="10")
        control_frame.pack(fill='x', pady=10)

        ttk.Button(control_frame, text="Load Audio File", command=self.load_audio).pack(side='left', padx=5)

        # ── Filter Type Selection ────────────────────────────────────────
        filter_type_frame = ttk.Frame(control_frame)
        filter_type_frame.pack(side='left', padx=20)

        ttk.Label(filter_type_frame, text="Filter Type:").pack(side='left', padx=5)
        self.audio_filter_type = ttk.Combobox(filter_type_frame, width=15, state='readonly',
                                              values=["Lowpass", "Highpass", "Bandpass", "Bandstop"])
        self.audio_filter_type.current(0)
        self.audio_filter_type.pack(side='left', padx=5)

        tk.Button(control_frame, text="🗑️ Delete", command=self.delete_audio,
                  fg="red", bg="#f0f0f0", relief='raised', cursor="hand2").pack(side='left', padx=5)

        # ── Dynamic Filter Options Panel ─────────────────────────────────
        self.audio_options_frame = ttk.LabelFrame(
            frame, text="Filter Options  —  select a filter type above", padding="8")
        self.audio_options_frame.pack(fill='x', pady=(0, 4))

        ttk.Label(self.audio_options_frame,
                  text="Select filter type and configure parameters below.",
                  foreground="gray").pack(anchor='w')

        # ── Build initial filter options (Lowpass) ──────────────────────
        self.build_audio_filter_options()

        # Bind combobox change
        self.audio_filter_type.bind('<<ComboboxSelected>>', lambda e: self.build_audio_filter_options())

        # ── Matplotlib canvas ────────────────────────────────────────────
        self.audio_fig = Figure(figsize=(10, 5), dpi=80)
        self.audio_canvas = FigureCanvasTkAgg(self.audio_fig, master=frame)
        self.audio_canvas.get_tk_widget().pack(fill='both', expand=True)

    def build_audio_filter_options(self):
        """Dynamically build filter options panel based on selected filter type."""
        for w in self.audio_options_frame.winfo_children():
            w.destroy()

        filter_type = self.audio_filter_type.get()
        self.audio_options_frame.config(text=f"Filter Options  —  {filter_type}")

        common_frame = ttk.Frame(self.audio_options_frame)
        common_frame.pack(fill='x', pady=5)

        ttk.Label(common_frame, text="Filter Design Method:").pack(side='left', padx=5)
        ttk.Combobox(common_frame, textvariable=self.audio_design_method, width=15, state='readonly',
                     values=["Butterworth", "Chebyshev I", "Bessel"]
                     ).pack(side='left', padx=5)

        ttk.Label(common_frame, text="Filter Order (1-10):").pack(side='left', padx=5)
        ttk.Spinbox(common_frame, from_=1, to=10, textvariable=self.audio_filter_order,
                    width=5).pack(side='left', padx=5)

        options_frame = ttk.Frame(self.audio_options_frame)
        options_frame.pack(fill='x', pady=5)

        if filter_type in ["Lowpass", "Highpass"]:
            ttk.Label(options_frame, text="Cutoff Frequency (Hz):").pack(side='left', padx=5)
            ttk.Spinbox(options_frame, from_=1, to=22050, increment=100,
                        textvariable=self.audio_cutoff, width=10
                        ).pack(side='left', padx=5)
            ttk.Label(options_frame, text="(Nyquist check enabled)", foreground="gray").pack(side='left', padx=5)

        elif filter_type in ["Bandpass", "Bandstop"]:
            ttk.Label(options_frame, text="Low Cutoff (Hz):").pack(side='left', padx=5)
            ttk.Spinbox(options_frame, from_=1, to=22050, increment=100,
                        textvariable=self.audio_cutoff_low, width=10
                        ).pack(side='left', padx=5)

            ttk.Label(options_frame, text="High Cutoff (Hz):").pack(side='left', padx=5)
            ttk.Spinbox(options_frame, from_=1, to=22050, increment=100,
                        textvariable=self.audio_cutoff_high, width=10
                        ).pack(side='left', padx=5)

        ttk.Button(options_frame, text="▶ Apply Filter",
                   command=self.apply_audio_filter).pack(side='left', padx=10)

        info_text = (
            f"Type: {filter_type} | "
            f"Method: {self.audio_design_method.get()} | "
            f"Order: {self.audio_filter_order.get()}"
        )
        ttk.Label(self.audio_options_frame, text=info_text, foreground="gray").pack(anchor='w', padx=5)

    def delete_audio(self):
        self.audio_data = None
        self.audio_fs = None
        self.filtered_audio = None
        self.audio_fig.clear()
        self.audio_canvas.draw()
        messagebox.showinfo("Success", "Audio data cleared!")

    def load_audio(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.audio_data, self.audio_fs = sf.read(filepath)
                if self.audio_data.ndim > 1:
                    self.audio_data = np.mean(self.audio_data, axis=1)
                self.plot_audio(self.audio_data, "Original Audio Signal")
                messagebox.showinfo("Success",
                    f"Loaded: {os.path.basename(filepath)}\nSample Rate: {self.audio_fs} Hz")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load audio: {str(e)}")

    def apply_audio_filter(self):
        """Apply the selected filter with customizable parameters."""
        if self.audio_data is None:
            messagebox.showwarning("Warning", "Please load an audio file first")
            return

        try:
            filter_type = self.audio_filter_type.get()
            design_method = self.audio_design_method.get()
            order = int(self.audio_filter_order.get())
            nyq = 0.5 * self.audio_fs

            if order < 1 or order > 10:
                raise ValueError("Filter order must be between 1 and 10.")

            if filter_type == "Lowpass":
                cutoff = float(self.audio_cutoff.get())
                if cutoff >= nyq:
                    raise ValueError(f"Cutoff frequency ({cutoff} Hz) must be less than Nyquist frequency ({nyq} Hz)")
                norm_cutoff = cutoff / nyq
                b, a = self._design_filter(design_method, order, norm_cutoff, 'low')
                title = f"Lowpass Filtered ({cutoff} Hz)"

            elif filter_type == "Highpass":
                cutoff = float(self.audio_cutoff.get())
                if cutoff >= nyq:
                    raise ValueError(f"Cutoff frequency ({cutoff} Hz) must be less than Nyquist frequency ({nyq} Hz)")
                norm_cutoff = cutoff / nyq
                b, a = self._design_filter(design_method, order, norm_cutoff, 'high')
                title = f"Highpass Filtered ({cutoff} Hz)"

            elif filter_type == "Bandpass":
                cutoff_low = float(self.audio_cutoff_low.get())
                cutoff_high = float(self.audio_cutoff_high.get())
                if cutoff_low >= cutoff_high:
                    raise ValueError("Low cutoff frequency must be less than high cutoff frequency.")
                if cutoff_high >= nyq:
                    raise ValueError(f"High cutoff ({cutoff_high} Hz) must be less than Nyquist ({nyq} Hz)")
                norm_cutoffs = [cutoff_low / nyq, cutoff_high / nyq]
                b, a = self._design_filter(design_method, order, norm_cutoffs, 'band')
                title = f"Bandpass Filtered ({cutoff_low}-{cutoff_high} Hz)"

            elif filter_type == "Bandstop":
                cutoff_low = float(self.audio_cutoff_low.get())
                cutoff_high = float(self.audio_cutoff_high.get())
                if cutoff_low >= cutoff_high:
                    raise ValueError("Low cutoff frequency must be less than high cutoff frequency.")
                if cutoff_high >= nyq:
                    raise ValueError(f"High cutoff ({cutoff_high} Hz) must be less than Nyquist ({nyq} Hz)")
                norm_cutoffs = [cutoff_low / nyq, cutoff_high / nyq]
                b, a = self._design_filter(design_method, order, norm_cutoffs, 'bandstop')
                title = f"Bandstop Filtered ({cutoff_low}-{cutoff_high} Hz)"

            self.filtered_audio = lfilter(b, a, self.audio_data)
            self.plot_audio_comparison(self.audio_data, self.filtered_audio, title)

            output_file = f"{filter_type.lower()}_filtered.wav"
            sf.write(output_file, self.filtered_audio, self.audio_fs)
            messagebox.showinfo("Success", f"Filtered audio saved as {output_file}")

        except (ValueError, TypeError) as e:
            messagebox.showerror("Error", f"Filter parameter error:\n{str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error during filtering:\n{str(e)}")

    def _design_filter(self, method, order, cutoff, btype):
        """Design filter using specified method."""
        try:
            if method == "Butterworth":
                return butter(order, cutoff, btype=btype)
            elif method == "Chebyshev I":
                return cheby1(order, rp=5, Wn=cutoff, btype=btype)
            elif method == "Bessel":
                return bessel(order, Wn=cutoff, btype=btype)
            else:
                raise ValueError(f"Unknown filter design method: {method}")
        except (TypeError, ValueError) as e:
            raise TypeError(f"Filter design error ({method}):\n{str(e)}")

    def plot_audio(self, signal, title):
        """Plot single audio signal."""
        self.audio_fig.clear()
        ax = self.audio_fig.add_subplot(111)
        ax.plot(signal, linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        self.audio_fig.tight_layout()
        self.audio_canvas.draw()

    def plot_audio_comparison(self, original, filtered, title):
        """Plot original vs filtered audio side-by-side."""
        self.audio_fig.clear()

        ax1 = self.audio_fig.add_subplot(211)
        ax1.plot(original, linewidth=0.5, color='blue')
        ax1.set_title("Original Audio Signal")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)

        ax2 = self.audio_fig.add_subplot(212)
        ax2.plot(filtered, linewidth=0.5, color='green')
        ax2.set_title(title)
        ax2.set_xlabel("Samples")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True, alpha=0.3)

        self.audio_fig.suptitle(
            f"Audio Filtering Comparison  |  fs={self.audio_fs} Hz  |  Order={self.audio_filter_order.get()}",
            fontsize=11, fontweight='bold')
        self.audio_fig.tight_layout()
        self.audio_canvas.draw()

    # ==================== TAB 3: Z-TRANSFORM ====================
    def create_z_transform_tab(self):
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Ω Lab 4: Z-Transform")

        info_frame = ttk.LabelFrame(frame, text="Instructions", padding="10")
        info_frame.pack(fill='x', pady=10)
        info_text = """Enter the sequence values as space-separated numbers and specify the starting index n.
Example 1: Sequence = "5 3 -3 0 4 -2", n = -2  (x(-2)=5, x(-1)=3, x(0)=-3, x(1)=0, x(2)=4, x(3)=-2)
Example 2: Sequence = "1 2 0 3 4 5", n = 0     (x(0)=1, x(1)=2, x(2)=0, x(3)=3, x(4)=4, x(5)=5)

The Z-transform formula: X(z) = Σ x(n)·z^(-n)
For DISCRETE TIME signals where n represents the sample number."""
        ttk.Label(info_frame, text=info_text, justify='left').pack(fill='x')

        input_frame = ttk.LabelFrame(frame, text="Z-Transform Calculator", padding="10")
        input_frame.pack(fill='x', pady=10)

        ttk.Label(input_frame, text="Sequence (space-separated):").pack(side='left', padx=5)
        self.z_input = ttk.Entry(input_frame, width=30)
        self.z_input.pack(side='left', padx=5)
        ttk.Label(input_frame, text="Starting Index n:").pack(side='left', padx=5)
        self.z_start_index = ttk.Entry(input_frame, width=8)
        self.z_start_index.insert(0, "0")
        self.z_start_index.pack(side='left', padx=5)
        ttk.Button(input_frame, text="Calculate Z-Transform",
                   command=self.calculate_z_transform).pack(side='left', padx=5)
        tk.Button(input_frame, text="🗑️ Delete", command=self.delete_z_transform,
                  fg="red", bg="#f0f0f0", relief='raised', cursor="hand2").pack(side='left', padx=5)

        output_frame = ttk.LabelFrame(frame, text="Result", padding="10")
        output_frame.pack(fill='both', expand=True, pady=10)
        self.z_output = tk.Text(output_frame, height=15, width=80, font=("Courier", 10))
        self.z_output.pack(fill='both', expand=True)

    def delete_z_transform(self):
        self.z_input.delete(0, tk.END)
        self.z_start_index.delete(0, tk.END)
        self.z_start_index.insert(0, "0")
        self.z_output.delete(1.0, tk.END)
        messagebox.showinfo("Success", "Z-Transform data cleared!")

    def calculate_z_transform(self):
        try:
            input_str = self.z_input.get().strip()
            if not input_str:
                messagebox.showwarning("Warning", "Please enter a sequence")
                return
            sequence = [int(x) for x in input_str.split()]
            n_start = int(self.z_start_index.get())
            result, detailed = self.z_transform_detailed(sequence, n_start)
            self.z_output.delete(1.0, tk.END)
            self.z_output.insert(tk.END, f"Input Sequence: {sequence}\n")
            self.z_output.insert(tk.END, f"Starting Index (n): {n_start}\n\n")
            self.z_output.insert(tk.END, f"Detailed Calculation:\n")
            self.z_output.insert(tk.END, f"{detailed}\n\n")
            self.z_output.insert(tk.END, f"Z-Transform Result:\n")
            self.z_output.insert(tk.END, f"X(z) = {result}\n")
        except ValueError:
            messagebox.showerror("Error", "Invalid input! Enter only numbers separated by spaces.")

    def z_transform_detailed(self, sequence, n_start=0):
        terms = []
        equation_steps = []

        equation_steps.append("Formula: X(z) = sum of x(n)*z^(-n) for all n")
        equation_steps.append("")
        equation_steps.append("Step-by-step computation:")
        equation_steps.append("-" * 50)

        for idx, val in enumerate(sequence):
            n = n_start + idx
            power = -n
            # Format z power label cleanly
            if power >= 0:
                z_label = f"z^{power}"
            else:
                z_label = f"z^({power})"

            # Always show every term including zeros
            equation_steps.append(f"  n={n:3d}:  x({n}) = {val:5}  ->  {val} * {z_label}")

            if val == 0:
                continue  # zero contributes nothing to X(z)

            if val == 1:
                term_str = z_label
            elif val == -1:
                term_str = f"-{z_label}"
            else:
                term_str = f"{val}*{z_label}"
            terms.append(term_str)

            # Show running accumulated X(z) after each non-zero term
            running = " + ".join(terms).replace("+ -", "- ")
            equation_steps.append(f"         X(z) so far = {running}")

        equation_steps.append("-" * 50)

        if not terms:
            return "0", "0"

        result = " + ".join(terms).replace("+ -", "- ")
        detailed = "\n".join(equation_steps)
        return result, detailed

    # ==================== TAB 4: INVERSE Z-TRANSFORM (FIXED) ====================
    def create_inverse_z_transform_tab(self):
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Ω Lab 4B: Inverse Z")

        info_frame = ttk.LabelFrame(frame, text="Instructions", padding="10")
        info_frame.pack(fill='x', pady=10)
        info_text = """Enter the Z-transform expression to find the inverse (recover x[n]).
IMPORTANT: Use the format: coefficient*z^power (with asterisk *)
Examples:
  • 1*z^0 + 2*z^-1 + 3*z^-2
  • 5*z^2 + 3*z^1 - 3*z^0 + 0*z^-1 + 4*z^-2
  • 4*z^4 - 1*z^3 - 3*z^1 + 4*z^0 + 3*z^-1

Specify the starting index n to determine where the sequence begins.
Example: If starting n = -2, the first value maps to the HIGHEST power term.

The Inverse Z-Transform recovers the discrete-time signal x[n] from X(z).
Powers are sorted DESCENDING: highest z-power → n_start, next → n_start+1, etc."""
        ttk.Label(info_frame, text=info_text, justify='left').pack(fill='x')

        input_frame = ttk.LabelFrame(frame, text="Inverse Z-Transform Calculator", padding="10")
        input_frame.pack(fill='x', pady=10)
        ttk.Label(input_frame, text="X(z) =").pack(side='left', padx=5)
        self.inverse_z_input = ttk.Entry(input_frame, width=60)
        self.inverse_z_input.pack(side='left', padx=5)
        ttk.Label(input_frame, text="Starting n:").pack(side='left', padx=5)
        self.inverse_z_start = ttk.Entry(input_frame, width=8)
        self.inverse_z_start.insert(0, "0")
        self.inverse_z_start.pack(side='left', padx=5)
        ttk.Button(input_frame, text="Calculate Inverse",
                   command=self.calculate_inverse_z_transform).pack(side='left', padx=5)
        tk.Button(input_frame, text="🗑️ Delete", command=self.delete_inverse_z_transform,
                  fg="red", bg="#f0f0f0", relief='raised', cursor="hand2").pack(side='left', padx=5)

        output_frame = ttk.LabelFrame(frame, text="Result", padding="10")
        output_frame.pack(fill='both', expand=True, pady=10)
        self.inverse_z_output = tk.Text(output_frame, height=15, width=80, font=("Courier", 10))
        self.inverse_z_output.pack(fill='both', expand=True)

    def delete_inverse_z_transform(self):
        self.inverse_z_input.delete(0, tk.END)
        self.inverse_z_start.delete(0, tk.END)
        self.inverse_z_start.insert(0, "0")
        self.inverse_z_output.delete(1.0, tk.END)
        messagebox.showinfo("Success", "Inverse Z-Transform data cleared!")

    def calculate_inverse_z_transform(self):
        try:
            input_str = self.inverse_z_input.get().strip()
            if not input_str:
                messagebox.showwarning("Warning", "Please enter a Z-transform expression")
                return
            n_start = int(self.inverse_z_start.get())
            coefficients = self.parse_z_transform_fixed(input_str)
            if not coefficients:
                messagebox.showerror("Error",
                    "Could not parse the expression. Check the format.\n"
                    "Use format: coeff*z^power  e.g.  5*z^2 + 3*z^1 - 3*z^0 + 0*z^-1 + 4*z^-2")
                return

            self.inverse_z_output.delete(1.0, tk.END)

            # ── Header ────────────────────────────────────────────────────
            self.inverse_z_output.insert(tk.END, f"Input X(z): {input_str}\n")
            self.inverse_z_output.insert(tk.END, f"Starting index n = {n_start}\n")
            self.inverse_z_output.insert(tk.END, "=" * 55 + "\n\n")

            # ── Sort powers DESCENDING: highest power = n_start ───────────
            sorted_powers = sorted(coefficients.keys(), reverse=True)
            x_values = [coefficients[p] for p in sorted_powers]

            # ── Step-by-step solution ─────────────────────────────────────
            self.inverse_z_output.insert(tk.END,
                "Principle: coefficient of z^k  ->  x[n] at that sample\n")
            self.inverse_z_output.insert(tk.END,
                "Powers sorted highest→lowest map to n_start, n_start+1, ...\n")
            self.inverse_z_output.insert(tk.END, "-" * 55 + "\n")

            for idx, power in enumerate(sorted_powers):
                n = n_start + idx
                coeff = coefficients[power]
                # Format power label
                p_label = f"z^{power}" if power >= 0 else f"z^({power})"
                zero_note = "  <- zero term (kept in sequence)" if coeff == 0 else ""
                self.inverse_z_output.insert(tk.END,
                    f"  {p_label}  ->  n={n:3d}  :  x[{n}] = {coeff}{zero_note}\n")

            self.inverse_z_output.insert(tk.END, "-" * 55 + "\n\n")

            # ── Final answer ──────────────────────────────────────────────
            self.inverse_z_output.insert(tk.END, "Sequence x[n]:\n")
            self.inverse_z_output.insert(tk.END,
                f"  {x_values}  (starting at n = {n_start})\n\n")
            self.inverse_z_output.insert(tk.END, "Final Answer:\n")
            self.inverse_z_output.insert(tk.END,
                f"  x[n] = {{ {', '.join(map(str, x_values))} }}\n")

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input. {str(e)}")

    def parse_z_transform_fixed(self, expr):
        """
        Parse Z-transform expression with format: coeff*z^power

        Uses re.findall with a signed-token pattern so that every term —
        including zero coefficients and negative coefficients after a minus
        sign — is captured correctly without any string-splitting hacks.
        """
        coefficients = {}
        # Remove all whitespace for clean matching
        expr = expr.replace(' ', '')
        # Pattern: optional leading sign, integer or decimal coeff, *z^, signed integer power
        pattern = re.compile(r'([+-]?\d+(?:\.\d+)?)\*z\^([+-]?\d+)')
        matches = pattern.findall(expr)
        for coeff_str, power_str in matches:
            coeff = float(coeff_str) if '.' in coeff_str else int(coeff_str)
            power = int(power_str)
            coefficients[power] = coeff
        return coefficients

    # ==================== TAB: LAB 5 - DFT & MOTOR ANALYSIS ====================
    def create_dft_lab5_tab(self):
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="⚙️ Lab 5: DFT Analysis")

        # --- Manual DFT Calculation Section (Ref: Page 6-7 of PDF) ---
        calc_frame = ttk.LabelFrame(frame, text="Manual DFT Calculator", padding="10")
        calc_frame.pack(fill='x', pady=(0, 5))

        row1 = ttk.Frame(calc_frame)
        row1.pack(fill='x', pady=2)
        ttk.Label(row1, text="Sequence x[n] (space separated):").pack(side='left', padx=5)
        self.lab5_seq_input = ttk.Entry(row1, width=30)
        self.lab5_seq_input.insert(0, "1 1 0 0")
        self.lab5_seq_input.pack(side='left', padx=5)

        ttk.Button(row1, text="Calculate DFT", command=self.lab5_calculate_dft).pack(side='left', padx=5)
        ttk.Button(row1, text="Show Twiddle Factors", command=self.lab5_show_twiddle).pack(side='left', padx=5)

        # --- Motor Vibration Simulation Section (Ref: Page 3-4 of PDF) ---
        motor_frame = ttk.LabelFrame(frame, text="Motor Vibration Case Study (Spectral Analysis)", padding="10")
        motor_frame.pack(fill='x', pady=(0, 5))

        ttk.Label(motor_frame, text="Select Condition:").pack(side='left', padx=5)
        self.lab5_motor_cond = ttk.Combobox(motor_frame, values=["New Installation (Normal)", "Worn Drive Gear (Defective)"], state="readonly", width=30)
        self.lab5_motor_cond.current(0)
        self.lab5_motor_cond.pack(side='left', padx=5)
        ttk.Button(motor_frame, text="Simulate Vibration Spectrum", command=self.lab5_simulate_motor).pack(side='left', padx=5)

        # --- Display Area ---
        display_container = ttk.Frame(frame)
        display_container.pack(fill='both', expand=True)

        self.lab5_txt = tk.Text(display_container, width=45, font=("Courier", 9), bg="#ffffff")
        self.lab5_txt.pack(side='left', fill='both', expand=False, padx=(0,5))

        self.lab5_fig = Figure(figsize=(7, 4), dpi=80)
        self.lab5_canvas = FigureCanvasTkAgg(self.lab5_fig, master=display_container)
        self.lab5_canvas.get_tk_widget().pack(side='right', fill='both', expand=True)

    def lab5_calculate_dft(self):
        try:
            raw_input = self.lab5_seq_input.get().strip()
            x = [float(val) for val in raw_input.split()]
            N = len(x)
            self.lab5_txt.delete(1.0, tk.END)
            self.lab5_txt.insert(tk.END, f"--- MANUAL DFT (N={N}) ---\n")

            X_results = []
            for k in range(N):
                re, im = 0, 0
                self.lab5_txt.insert(tk.END, f"\nK = {k}:\n")
                for n in range(N):
                    angle = 2 * np.pi * k * n / N
                    term_re = x[n] * np.cos(angle)
                    term_im = -x[n] * np.sin(angle)
                    re += term_re
                    im += term_im
                    # Displaying steps like the PDF
                    sign = "-" if term_im >= 0 else "+" # showing negative j for positive angle
                    self.lab5_txt.insert(tk.END, f" n={n}: {x[n]} * (cos({angle:.2f}) - j sin({angle:.2f}))\n")

                X_results.append(complex(re, im))
                self.lab5_txt.insert(tk.END, f" Result X[{k}] = {re:.2f} + ({im:.2f})j\n")

            # --- Final Combined Result in Curly Braces (Ref: Page 7) ---
            formatted_list = []
            for val in X_results:
                r = round(val.real, 2)
                i = round(val.imag, 2)
                # Clean formatting (e.g., 1.0 - 1.0j)
                if i == 0: formatted_list.append(f"{r}")
                elif i > 0: formatted_list.append(f"{r}+{i}j")
                else: formatted_list.append(f"{r}{i}j")

            final_res_str = ", ".join(formatted_list)
            self.lab5_txt.insert(tk.END, "\n" + "="*40 + "\n")
            self.lab5_txt.insert(tk.END, f"FINAL RESULT:\n")
            self.lab5_txt.insert(tk.END, f"X(k) = {{ {final_res_str} }}\n")
            self.lab5_txt.insert(tk.END, "="*40 + "\n")

            # Plot Magnitude
            self.lab5_fig.clear()
            ax = self.lab5_fig.add_subplot(111)
            ax.stem(range(N), [np.abs(val) for val in X_results], basefmt=" ")
            ax.set_title("DFT Magnitude Spectrum |X[k]|")
            ax.set_xlabel("Frequency Index (k)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            self.lab5_fig.tight_layout()
            self.lab5_canvas.draw()

        except ValueError:
            messagebox.showerror("Error", "Invalid input sequence. Use numbers separated by spaces.")

    def lab5_show_twiddle(self):
        try:
            N = len(self.lab5_seq_input.get().split())
            self.lab5_txt.delete(1.0, tk.END)
            self.lab5_txt.insert(tk.END, f"--- TWIDDLE FACTORS (W_N) for N={N} ---\n")
            self.lab5_txt.insert(tk.END, "Formula: W_N = e^(-j2π/N)\n\n")
            for k in range(N):
                val = np.exp(-2j * np.pi * k / N)
                self.lab5_txt.insert(tk.END, f"W_{N}^{k} = {val.real:.3f} + ({val.imag:.3f})j\n")
        except:
            pass

    def lab5_simulate_motor(self):
        fs = 1000
        t = np.linspace(0, 1, fs)
        condition = self.lab5_motor_cond.get()
        sig_shaft = 1.0 * np.sin(2 * np.pi * 20 * t)

        if "New" in condition:
            sig_gear = 0.2 * np.sin(2 * np.pi * 240 * t)
            status_color = "blue"
        else:
            sig_gear = 1.5 * np.sin(2 * np.pi * 240 * t)
            status_color = "red"

        noise = 0.3 * np.random.normal(size=fs)
        total_signal = sig_shaft + sig_gear + noise
        freqs = np.fft.rfftfreq(fs, 1/fs)
        mags = np.abs(np.fft.rfft(total_signal))

        self.lab5_fig.clear()
        ax = self.lab5_fig.add_subplot(111)
        ax.plot(freqs, mags, color=status_color)
        ax.set_xlim(0, 350)
        ax.set_title(f"Motor Vibration Spectrum: {condition}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Spectral Amplitude")
        ax.annotate('Motor Shaft (20Hz)', xy=(20, mags[20]), xytext=(40, max(mags)*0.9),
                    arrowprops=dict(facecolor='black', arrowstyle='->'))
        ax.annotate('Drive Gear (240Hz)', xy=(240, mags[240]), xytext=(250, mags[240]),
                    arrowprops=dict(facecolor='black', arrowstyle='->'))
        ax.grid(True, alpha=0.3)
        self.lab5_fig.tight_layout()
        self.lab5_canvas.draw()

    # ==================== TAB 5: FFT ALGORITHM VISUALIZER (LAB 6) ====================
    def create_fft_tab(self):
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="⚡ Lab 6: FFT vs DFT")

        # --- Top Section: Algorithm Info ---
        top_frame = ttk.Frame(frame)
        top_frame.pack(fill='x', pady=5)

        ctrl_frame = ttk.LabelFrame(top_frame, text="Race Configuration", padding="10")
        ctrl_frame.pack(side='left', fill='y', padx=5)

        ttk.Label(ctrl_frame, text="Signal Size (N):").pack(anchor='w')
        self.fft_n_size = ttk.Combobox(ctrl_frame, values=["64", "128", "256", "512"], width=10, state="readonly")
        self.fft_n_size.current(0) # Default 64
        self.fft_n_size.pack(pady=5)

        self.race_btn = ttk.Button(ctrl_frame, text="🚩 Start Algorithm Race", command=self.start_fft_race)
        self.race_btn.pack(fill='x', pady=2)

        tk.Button(ctrl_frame, text="🗑️ Clear", command=self.delete_fft,
                  fg="red", bg="#f0f0f0", relief='raised').pack(fill='x', pady=2)

        # --- The "How it Works" Scoreboard ---
        how_frame = ttk.LabelFrame(top_frame, text="Algorithm Comparison Logic", padding="10")
        how_frame.pack(side='left', fill='both', expand=True, padx=5)

        self.lbl_logic_dft = ttk.Label(how_frame, text="DFT: Brute Force Loop\nEvery sample x every frequency.",
                                       foreground="red", font=("Courier", 9))
        self.lbl_logic_dft.pack(anchor='w', pady=2)

        self.lbl_logic_fft = ttk.Label(how_frame, text="FFT: Divide & Conquer (Butterfly)\nSplits signal into Even/Odd recursively.",
                                       foreground="green", font=("Courier", 9))
        self.lbl_logic_fft.pack(anchor='w', pady=2)

        self.lbl_math_stats = ttk.Label(how_frame, text="Math Ops: ---", font=("Helvetica", 10, "bold"), foreground="blue")
        self.lbl_math_stats.pack(pady=5)

        # --- Plots ---
        self.fft_fig = Figure(figsize=(10, 5), dpi=80)
        self.fft_canvas = FigureCanvasTkAgg(self.fft_fig, master=frame)
        self.fft_canvas.get_tk_widget().pack(fill='both', expand=True)

        self.is_racing = False

    def delete_fft(self):
        self.is_racing = False
        self.fft_fig.clear()
        self.fft_canvas.draw()
        self.lbl_math_stats.config(text="Math Ops: ---")
        self.race_btn.config(state="normal")

    def start_fft_race(self):
        if self.is_racing: return
        self.is_racing = True
        self.race_btn.config(state="disabled")

        N = int(self.fft_n_size.get())
        fs = 1000
        t = np.arange(N) / fs
        # Signal based on PDF Motor Study: 20Hz and 240Hz
        x = np.sin(2 * np.pi * 20 * t) + 0.5 * np.sin(2 * np.pi * 240 * t)

        self.fft_fig.clear()
        self.ax_dft = self.fft_fig.add_subplot(121)
        self.ax_fft = self.fft_fig.add_subplot(122)

        # 1. SHOW THE FFT LOGIC (Divide & Conquer)
        t0 = time.perf_counter()
        X_fft = np.fft.fft(x)
        fft_time = (time.perf_counter() - t0) * 1000

        # FFT Math: N * log2(N)
        fft_ops = int(N * np.log2(N))
        dft_ops = N * N

        self.lbl_math_stats.config(text=f"For N={N}:\nDFT needs {dft_ops} operations\nFFT only needs {fft_ops} operations!")

        freqs = np.fft.fftfreq(N, 1/fs)
        self.ax_fft.stem(freqs[:N//2], np.abs(X_fft[:N//2]), linefmt='g-', markerfmt='go', basefmt=' ')
        self.ax_fft.set_title(f"FFT Algorithm\n{int(np.log2(N))} 'Butterfly' Stages")
        self.fft_canvas.draw()

        # 2. START THE SLOW DFT ANIMATION
        self.dft_results = np.zeros(N, dtype=complex)
        self.dft_start_clock = time.perf_counter()
        self._animate_dft_logic(x, 0, N, fs)

    def _animate_dft_logic(self, x, k, N, fs):
        if not self.is_racing: return

        if k >= N // 2:
            total_time = (time.perf_counter() - self.dft_start_clock) * 1000
            self.lbl_math_stats.config(text=f"RACE FINISHED\nDFT: {total_time:.1f}ms\nFFT: Instant")
            self.race_btn.config(state="normal")
            self.is_racing = False
            return

        # Manual Summation (The way shown on Page 5 of your PDF)
        # We calculate EACH bin using a loop.
        for n in range(N):
            theta = 2 * np.pi * k * n / N
            self.dft_results[k] += x[n] * (np.cos(theta) - 1j * np.sin(theta))

        # Update Plot
        self.ax_dft.clear()
        self.ax_dft.set_title(f"Manual DFT Loop\nBin {k} uses {N} multiplications")
        freqs = np.fft.fftfreq(N, 1/fs)
        self.ax_dft.stem(freqs[:k+1], np.abs(self.dft_results[:k+1]), linefmt='r-', markerfmt='ro', basefmt=' ')
        self.ax_dft.set_xlim(0, 500)
        self.ax_dft.set_ylim(0, N/2 + 5)

        self.fft_canvas.draw()

        # The delay represents the extra time the CPU takes for O(N^2)
        self.root.after(15, lambda: self._animate_dft_logic(x, k + 1, N, fs))

    # ==================== TAB 6: WINDOWING ====================
    def create_windowing_tab(self):
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="🪟 Lab 7: Windowing")

        # --- Top Container for Parameters ---
        top_frame = ttk.LabelFrame(frame, text="Window Parameters", padding="10")
        top_frame.pack(side='top', fill='x', pady=(0, 10))

        # First row: Numerical Inputs (N, f1, f2, A2)
        input_row = ttk.Frame(top_frame)
        input_row.pack(side='top', fill='x', pady=2)

        ttk.Label(input_row, text="Signal Length (N):").pack(side='left', padx=(5, 2))
        self.window_N = ttk.Entry(input_row, width=8)
        self.window_N.insert(0, "64")
        self.window_N.pack(side='left', padx=(0, 15))

        ttk.Label(input_row, text="Frequency 1 (f1):").pack(side='left', padx=(5, 2))
        self.window_f1 = ttk.Entry(input_row, width=8)
        self.window_f1.insert(0, "5")
        self.window_f1.pack(side='left', padx=(0, 15))

        ttk.Label(input_row, text="Frequency 2 (f2):").pack(side='left', padx=(5, 2))
        self.window_f2 = ttk.Entry(input_row, width=8)
        self.window_f2.insert(0, "15")
        self.window_f2.pack(side='left', padx=(0, 15))

        ttk.Label(input_row, text="Amplitude 2 (A2):").pack(side='left', padx=(5, 2))
        self.window_A2 = ttk.Entry(input_row, width=8)
        self.window_A2.insert(0, "0.5")
        self.window_A2.pack(side='left', padx=(0, 15))

        # Second row: Window Buttons and Delete
        button_row = ttk.Frame(top_frame)
        button_row.pack(side='top', fill='x', pady=5)

        ttk.Label(button_row, text="Select Window:").pack(side='left', padx=(5, 5))
        ttk.Button(button_row, text="Rectangular", command=lambda: self.apply_window('Rectangular')).pack(side='left', padx=2)
        ttk.Button(button_row, text="Blackman", command=lambda: self.apply_window('Blackman')).pack(side='left', padx=2)
        ttk.Button(button_row, text="Hann", command=lambda: self.apply_window('Hann')).pack(side='left', padx=2)
        ttk.Button(button_row, text="Hamming", command=lambda: self.apply_window('Hamming')).pack(side='left', padx=2)

        tk.Button(button_row, text="🗑️ Delete", command=self.delete_windowing,
                  fg="red", bg="#f0f0f0", relief='raised', cursor="hand2").pack(side='right', padx=10)

        # --- Bottom Container for Graphs ---
        graph_container = ttk.Frame(frame)
        graph_container.pack(side='top', fill='both', expand=True)

        self.window_fig = Figure(figsize=(14, 5), dpi=80)
        self.window_canvas = FigureCanvasTkAgg(self.window_fig, master=graph_container)
        self.window_canvas.get_tk_widget().pack(fill='both', expand=True)

        # Draw initial plot
        self.apply_window('Rectangular')

    def delete_windowing(self):
        self.window_N.delete(0, tk.END); self.window_N.insert(0, "64")
        self.window_f1.delete(0, tk.END); self.window_f1.insert(0, "5")
        self.window_f2.delete(0, tk.END); self.window_f2.insert(0, "15")
        self.window_A2.delete(0, tk.END); self.window_A2.insert(0, "0.5")
        self.apply_window('Rectangular')
        messagebox.showinfo("Success", "Windowing parameters reset!")

    def apply_window(self, window_name):
        try:
            N  = int(self.window_N.get())
            f1 = int(self.window_f1.get())
            f2 = int(self.window_f2.get())
            A2 = float(self.window_A2.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for all parameters")
            return
        n = np.arange(N)
        x = np.sin(2 * np.pi * f1 * n / N) + A2 * np.sin(2 * np.pi * f2 * n / N)
        windows = {
            "Rectangular": np.ones(N),
            "Blackman": np.blackman(N),
            "Hann": np.hanning(N),
            "Hamming": np.hamming(N)
        }
        w = windows[window_name]
        xw = x * w
        X  = np.fft.fft(x)
        Xw = np.fft.fft(xw)
        freq = np.arange(N)

        self.window_fig.clear()
        ax1 = self.window_fig.add_subplot(131)
        ax1.plot(n, x, 'b-', linewidth=2)
        ax1.set_title("Original Signal"); ax1.set_xlabel("n"); ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)

        ax2 = self.window_fig.add_subplot(132)
        ax2.plot(n, w, 'r--', linewidth=2, label='Window')
        ax2.plot(n, xw, 'g-', linewidth=2, label='Windowed Signal')
        ax2.set_title(f"{window_name} Window (Time Domain)")
        ax2.set_xlabel("n"); ax2.set_ylabel("Amplitude"); ax2.legend(); ax2.grid(True, alpha=0.3)

        ax3 = self.window_fig.add_subplot(133)
        ax3.plot(freq, np.abs(X),  'b-', label="Original",  linewidth=2)
        ax3.plot(freq, np.abs(Xw), 'r-', label="Windowed", linewidth=2)
        ax3.set_title("FFT Comparison")
        ax3.set_xlabel("Frequency Bin"); ax3.set_ylabel("Magnitude"); ax3.legend(); ax3.grid(True, alpha=0.3)

        self.window_fig.suptitle(
            f"{window_name} Window Analysis (N={N}, f1={f1}, f2={f2}, A2={A2})",
            fontsize=12, fontweight='bold')
        self.window_fig.tight_layout()
        self.window_canvas.draw()

    # ==================== TAB: LAB 8 - FIR & IIR FILTERS (UPDATED) ====================
    def create_fir_iir_lab8_tab(self):
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="⚖️ Lab 8: FIR vs IIR")

        # --- Controls Section ---
        ctrl_frame = ttk.LabelFrame(frame, text="Filter Design Parameters", padding="10")
        ctrl_frame.pack(fill='x', pady=(0, 5))

        row1 = ttk.Frame(ctrl_frame)
        row1.pack(fill='x', pady=2)

        ttk.Label(row1, text="Filter Type:").pack(side='left', padx=5)
        self.lab8_type = ttk.Combobox(row1, values=["Lowpass", "Highpass", "Bandpass", "Bandstop"], state="readonly", width=12)
        self.lab8_type.current(0)
        self.lab8_type.pack(side='left', padx=5)

        ttk.Label(row1, text="Order (N):").pack(side='left', padx=5)
        self.lab8_order = ttk.Spinbox(row1, from_=1, to=100, width=5)
        self.lab8_order.set(10)
        self.lab8_order.pack(side='left', padx=5)

        # Dual Cutoff Inputs
        ttk.Label(row1, text="Low Cut (Hz):").pack(side='left', padx=5)
        self.lab8_cutoff_low = ttk.Entry(row1, width=8)
        self.lab8_cutoff_low.insert(0, "500")
        self.lab8_cutoff_low.pack(side='left', padx=5)

        ttk.Label(row1, text="High Cut (Hz):").pack(side='left', padx=5)
        self.lab8_cutoff_high = ttk.Entry(row1, width=8)
        self.lab8_cutoff_high.insert(0, "1500")
        self.lab8_cutoff_high.pack(side='left', padx=5)

        ttk.Button(row1, text="Compare FIR vs IIR", command=self.lab8_compare_filters).pack(side='left', padx=20)

        # --- Information & Comparison Table ---
        info_frame = ttk.Frame(frame)
        info_frame.pack(fill='both', expand=True)

        self.lab8_fig = Figure(figsize=(7, 4), dpi=90)
        self.lab8_canvas = FigureCanvasTkAgg(self.lab8_fig, master=info_frame)
        self.lab8_canvas.get_tk_widget().pack(side='left', fill='both', expand=True)

        # Right side: Comparison Table (Ref: Page 13)
        table_frame = ttk.LabelFrame(info_frame, text="FIR vs IIR Traits (Ref: Page 13)", padding="10")
        table_frame.pack(side='right', fill='y', padx=10)

        traits = [
            ("Stability", "Always Stable", "Can be Unstable"),
            ("Phase", "Linear (Constant)", "Non-Linear"),
            ("Speed", "Slower", "Faster"),
            ("Complexity", "Lower (Easier)", "Higher"),
            ("Roll-off", "Gradual", "Sharper")
        ]

        ttk.Label(table_frame, text="Feature", font=('Helvetica', 9, 'bold')).grid(row=0, column=0, sticky='w', padx=5)
        ttk.Label(table_frame, text="FIR", font=('Helvetica', 9, 'bold'), foreground="blue").grid(row=0, column=1, sticky='w', padx=5)
        ttk.Label(table_frame, text="IIR", font=('Helvetica', 9, 'bold'), foreground="red").grid(row=0, column=2, sticky='w', padx=5)

        for i, (feat, fir, iir) in enumerate(traits):
            ttk.Label(table_frame, text=feat).grid(row=i+1, column=0, sticky='w', padx=5, pady=2)
            ttk.Label(table_frame, text=fir, foreground="gray").grid(row=i+1, column=1, sticky='w', padx=5)
            ttk.Label(table_frame, text=iir, foreground="gray").grid(row=i+1, column=2, sticky='w', padx=5)

        self.lab8_eqn = ttk.Label(table_frame, text="\nFIR: y[n] = Σ a(k)x[n-k]\nIIR: y[n] = Σ a(k)x[n-k] + Σ b(j)y[n-j]",
                                  justify='left', font=("Courier", 8))
        self.lab8_eqn.grid(row=len(traits)+1, column=0, columnspan=3, pady=10)

    def lab8_compare_filters(self):
        try:
            from scipy.signal import freqz, firwin
            fs = 8000
            nyq = fs / 2
            order = int(self.lab8_order.get())
            f_type = self.lab8_type.get()

            low = float(self.lab8_cutoff_low.get())
            high = float(self.lab8_cutoff_high.get())

            # Frequency Validation
            if low >= nyq or (f_type in ["Bandpass", "Bandstop"] and high >= nyq):
                messagebox.showerror("Error", f"Frequencies must be < Nyquist ({nyq}Hz)")
                return
            if f_type in ["Bandpass", "Bandstop"] and low >= high:
                messagebox.showerror("Error", "Low Cutoff must be less than High Cutoff")
                return

            # Design Logic
            if f_type == "Lowpass":
                b_iir, a_iir = butter(order, low/nyq, btype='low')
                b_fir = firwin(order + 1, low/nyq, pass_zero=True)
            elif f_type == "Highpass":
                b_iir, a_iir = butter(order, low/nyq, btype='high')
                b_fir = firwin(order + 1, low/nyq, pass_zero=False)
            elif f_type == "Bandpass":
                b_iir, a_iir = butter(order, [low/nyq, high/nyq], btype='bandpass')
                b_fir = firwin(order + 1, [low/nyq, high/nyq], pass_zero=False)
            elif f_type == "Bandstop":
                b_iir, a_iir = butter(order, [low/nyq, high/nyq], btype='bandstop')
                b_fir = firwin(order + 1, [low/nyq, high/nyq], pass_zero=True)

            # Frequency Response
            w_iir, h_iir = freqz(b_iir, a_iir, worN=2000)
            w_fir, h_fir = freqz(b_fir, [1], worN=2000)

            # Plotting
            self.lab8_fig.clear()
            ax = self.lab8_fig.add_subplot(111)
            ax.plot(w_iir * nyq / np.pi, 20 * np.log10(np.abs(h_iir)), 'r-', label=f'IIR (Order {order})')
            ax.plot(w_fir * nyq / np.pi, 20 * np.log10(np.abs(h_fir)), 'b--', label=f'FIR (Order {order})')

            ax.axvline(low, color='green', linestyle=':', label='Low Cutoff')
            if f_type in ["Bandpass", "Bandstop"]:
                ax.axvline(high, color='orange', linestyle=':', label='High Cutoff')

            ax.set_title(f"Comparison: {f_type}")
            ax.set_ylabel("Magnitude (dB)")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylim(-80, 5)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower left')

            self.lab8_fig.tight_layout()
            self.lab8_canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", str(e))


def main():
    root = tk.Tk()
    app = DSPMasterProgram(root)
    root.mainloop()


if __name__ == "__main__":
    main()
