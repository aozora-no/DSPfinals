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

class DSPMasterProgram:
    def __init__(self, root):
        self.root = root
        self.root.title("Digital Signal Processing Master Program")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_lab1_tab()
        self.create_image_processing_tab()
        self.create_audio_processing_tab()
        self.create_z_transform_tab()
        self.create_inverse_z_transform_tab()
        self.create_fft_tab()
        self.create_windowing_tab()
        self.create_sampling_tab()

    # ==================== TAB 0: LAB 1 - CUSTOM SIGNAL PLOTTER ====================
    def create_lab1_tab(self):
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Lab 1 - Signal Plotter")

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

    # ---------- helpers ----------
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
        self.notebook.add(frame, text="Image Processing (Lab 2)")

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
        self.notebook.add(frame, text="Audio Processing (Lab 3)")
        
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
        self.notebook.add(frame, text="Z-Transform (Lab 4)")
        
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
        self.notebook.add(frame, text="Inverse Z-Transform (Lab 4B)")
        
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

        Examples handled:
            5*z^2 + 3*z^1 - 3*z^0 + 0*z^-1 + 4*z^-2
            4*z^4 - 1*z^3 - 3*z^1 + 4*z^0 + 3*z^-1
            1*z^0 + 2*z^-1 + 3*z^-2
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
    
    # ==================== TAB 5: FFT ANALYSIS ====================
    def create_fft_tab(self):
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="FFT Analysis (Lab 6)")
        
        control_frame = ttk.LabelFrame(frame, text="FFT Controls", padding="10")
        control_frame.pack(fill='x', pady=10)
        ttk.Label(control_frame, text="Signal Example:").pack(side='left', padx=5)
        ttk.Button(control_frame, text="Generate Signal",
                   command=self.generate_fft_signal).pack(side='left', padx=5)
        tk.Button(control_frame, text="🗑️ Delete", command=self.delete_fft,
                  fg="red", bg="#f0f0f0", relief='raised', cursor="hand2").pack(side='left', padx=5)
        
        self.fft_fig = Figure(figsize=(10, 5), dpi=80)
        self.fft_canvas = FigureCanvasTkAgg(self.fft_fig, master=frame)
        self.fft_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def delete_fft(self):
        self.fft_fig.clear()
        self.fft_canvas.draw()
        messagebox.showinfo("Success", "FFT plot cleared!")
    
    def generate_fft_signal(self):
        x = np.array([-1, 5, 7, 2])
        N = len(x)
        X = np.fft.fft(x)
        freq = np.fft.fftfreq(N)
        magnitude = np.abs(X)
        
        self.fft_fig.clear()
        ax1 = self.fft_fig.add_subplot(121)
        ax1.stem(range(N), x, basefmt=' ')
        ax1.set_title("Input Signal x[n]")
        ax1.set_xlabel("n")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)
        
        ax2 = self.fft_fig.add_subplot(122)
        ax2.stem(freq, magnitude, basefmt=' ')
        ax2.set_title("Magnitude Spectrum |X[k]|")
        ax2.set_xlabel("Frequency")
        ax2.set_ylabel("Magnitude")
        ax2.grid(True, alpha=0.3)
        
        self.fft_fig.tight_layout()
        self.fft_canvas.draw()
    
    # ==================== TAB 6: WINDOWING ====================
    def create_windowing_tab(self):
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Windowing (Lab 7)")
        
        left_frame = ttk.LabelFrame(frame, text="Window Parameters", padding="10")
        left_frame.pack(side='left', fill='both', padx=10, pady=10)
        
        ttk.Label(left_frame, text="Signal Length (N):").pack(anchor='w', pady=5)
        self.window_N = ttk.Entry(left_frame, width=10)
        self.window_N.insert(0, "64")
        self.window_N.pack(anchor='w', pady=5)
        
        ttk.Label(left_frame, text="Frequency 1 (f1):").pack(anchor='w', pady=5)
        self.window_f1 = ttk.Entry(left_frame, width=10)
        self.window_f1.insert(0, "5")
        self.window_f1.pack(anchor='w', pady=5)
        
        ttk.Label(left_frame, text="Frequency 2 (f2):").pack(anchor='w', pady=5)
        self.window_f2 = ttk.Entry(left_frame, width=10)
        self.window_f2.insert(0, "15")
        self.window_f2.pack(anchor='w', pady=5)
        
        ttk.Label(left_frame, text="Amplitude 2 (A2):").pack(anchor='w', pady=5)
        self.window_A2 = ttk.Entry(left_frame, width=10)
        self.window_A2.insert(0, "0.5")
        self.window_A2.pack(anchor='w', pady=5)
        
        ttk.Label(left_frame, text="Select Window:").pack(anchor='w', pady=(20, 5))
        ttk.Button(left_frame, text="Rectangular",
                   command=lambda: self.apply_window('Rectangular')).pack(fill='x', pady=3)
        ttk.Button(left_frame, text="Blackman",
                   command=lambda: self.apply_window('Blackman')).pack(fill='x', pady=3)
        ttk.Button(left_frame, text="Hann",
                   command=lambda: self.apply_window('Hann')).pack(fill='x', pady=3)
        ttk.Button(left_frame, text="Hamming",
                   command=lambda: self.apply_window('Hamming')).pack(fill='x', pady=3)
        tk.Button(left_frame, text="🗑️ Delete", command=self.delete_windowing,
                  fg="red", bg="#f0f0f0", relief='raised', cursor="hand2").pack(fill='x', pady=10)
        
        right_frame = ttk.Frame(frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        self.window_fig = Figure(figsize=(14, 5), dpi=80)
        self.window_canvas = FigureCanvasTkAgg(self.window_fig, master=right_frame)
        self.window_canvas.get_tk_widget().pack(fill='both', expand=True)
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
    
    # ==================== TAB 7: SAMPLING ====================
    def create_sampling_tab(self):
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Sampling Theory (Labs 1,5)")
        
        control_frame = ttk.LabelFrame(frame, text="Sampling Examples", padding="10")
        control_frame.pack(fill='x', pady=10)
        ttk.Button(control_frame, text="Continuous Signal (1kHz)",
                   command=self.plot_first_task).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Sampling 1kHz @ 8kHz",
                   command=self.plot_second_task).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Sampling 2kHz @ 8kHz",
                   command=self.plot_third_task).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Sampling 6kHz @ 8kHz (Aliasing)",
                   command=self.plot_fourth_task).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Multiple Frequencies",
                   command=self.plot_fifth_task).pack(side='left', padx=5)
        tk.Button(control_frame, text="🗑️ Delete", command=self.delete_sampling,
                  fg="red", bg="#f0f0f0", relief='raised', cursor="hand2").pack(side='left', padx=5)
        
        self.sampling_fig = Figure(figsize=(10, 5), dpi=80)
        self.sampling_canvas = FigureCanvasTkAgg(self.sampling_fig, master=frame)
        self.sampling_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def delete_sampling(self):
        self.sampling_fig.clear()
        self.sampling_canvas.draw()
        messagebox.showinfo("Success", "Sampling plot cleared!")
    
    def plot_first_task(self):
        t = np.linspace(-1, 1, 1000)
        xt = np.cos(2 * np.pi * 1000 * t)
        self.sampling_fig.clear()
        ax = self.sampling_fig.add_subplot(111)
        ax.plot(t, xt)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
        ax.set_title("First Task: Continuous 1kHz Cosine Signal")
        ax.grid(True, alpha=0.3)
        self.sampling_fig.tight_layout(); self.sampling_canvas.draw()
    
    def plot_second_task(self):
        f, fs = 1000, 8000
        t  = np.linspace(0, 0.01, 10000)
        ts = np.arange(0, 0.01, 1 / fs)
        x_t      = np.cos(2 * np.pi * f * t)
        x_sample = np.cos(2 * np.pi * f * ts)
        self.sampling_fig.clear()
        ax = self.sampling_fig.add_subplot(111)
        ax.plot(t, x_t, label="Continuous Signal", linewidth=2)
        ax.stem(ts, x_sample, linefmt='r-', markerfmt='ro', basefmt="", label="Sampled (fs=8kHz)")
        ax.set_title("Sampling 1kHz at 8kHz (Nyquist Satisfied)")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); ax.legend(); ax.grid(True, alpha=0.3)
        self.sampling_fig.tight_layout(); self.sampling_canvas.draw()
    
    def plot_third_task(self):
        f, fs = 2000, 8000
        t  = np.linspace(0, 0.01, 10000)
        ts = np.arange(0, 0.01, 1 / fs)
        x_t      = np.cos(2 * np.pi * f * t)
        x_sample = np.cos(2 * np.pi * f * ts)
        self.sampling_fig.clear()
        ax = self.sampling_fig.add_subplot(111)
        ax.plot(t, x_t, label="Continuous Signal", linewidth=2)
        ax.stem(ts, x_sample, linefmt='r-', markerfmt='ro', basefmt="", label="Sampled (fs=8kHz)")
        ax.set_title("Sampling 2kHz at 8kHz (Nyquist Satisfied)")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); ax.legend(); ax.grid(True, alpha=0.3)
        self.sampling_fig.tight_layout(); self.sampling_canvas.draw()
    
    def plot_fourth_task(self):
        f, fs = 6000, 8000
        t  = np.linspace(0, 0.01, 10000)
        ts = np.arange(0, 0.01, 1 / fs)
        x_t      = np.cos(2 * np.pi * f * t)
        x_sample = np.cos(2 * np.pi * f * ts)
        self.sampling_fig.clear()
        ax = self.sampling_fig.add_subplot(111)
        ax.plot(t, x_t, label="Continuous Signal (6kHz)", linewidth=2)
        ax.stem(ts, x_sample, linefmt='r-', markerfmt='ro', basefmt="", label="Sampled (fs=8kHz)")
        ax.set_title("⚠️ Sampling 6kHz at 8kHz (ALIASING - Nyquist Violated!)")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); ax.legend(); ax.grid(True, alpha=0.3)
        self.sampling_fig.tight_layout(); self.sampling_canvas.draw()
    
    def plot_fifth_task(self):
        n = np.arange(0, 21)
        x1 = np.cos(0.2 * np.pi * n)
        x2 = np.cos(2.2 * np.pi * n)
        x3 = np.cos(4.2 * np.pi * n)
        self.sampling_fig.clear()
        ax = self.sampling_fig.add_subplot(111)
        ax.stem(n, x1, basefmt=' ', linefmt='b-', markerfmt='bo', label='0.2π')
        ax.stem(n, x2, basefmt=' ', linefmt='r-', markerfmt='rs', label='2.2π')
        ax.stem(n, x3, basefmt=' ', linefmt='g-', markerfmt='g^', label='4.2π')
        ax.set_title("Fifth Task: Multiple Frequencies Comparison")
        ax.set_xlabel("n (samples)"); ax.set_ylabel("Amplitude"); ax.legend(); ax.grid(True, alpha=0.3)
        self.sampling_fig.tight_layout(); self.sampling_canvas.draw()


def main():
    root = tk.Tk()
    app = DSPMasterProgram(root)
    root.mainloop()

if __name__ == "__main__":
    main()