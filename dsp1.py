import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import cv2
import soundfile as sf
from scipy.signal import butter, lfilter
import os
from PIL import Image, ImageTk
import io

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
        self.create_image_processing_tab()
        self.create_audio_processing_tab()
        self.create_z_transform_tab()
        self.create_fft_tab()
        self.create_windowing_tab()
        self.create_sampling_tab()
        
    # ==================== TAB 1: IMAGE PROCESSING ====================
    def create_image_processing_tab(self):
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Image Processing (Lab 2)")
        
        # Control Panel
        control_frame = ttk.LabelFrame(frame, text="Image Processing Controls", padding="10")
        control_frame.pack(fill='x', pady=10)
        
        ttk.Button(control_frame, text="Load Image", command=self.load_image).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Grayscale", command=self.apply_grayscale).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Binary (B&W)", command=self.apply_binary).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Median Blur", command=self.apply_median_blur).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Laplacian", command=self.apply_laplacian).pack(side='left', padx=5)
        
        self.image_label = ttk.Label(frame, text="Load an image to begin", background="white")
        self.image_label.pack(fill='both', expand=True, pady=10)
        
        self.original_image = None
        self.current_image = None
        
    def load_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.bmp"), ("All files", "*.*")]
        )
        if filepath:
            self.original_image = cv2.imread(filepath)
            self.current_image = self.original_image.copy()
            self.display_image(self.original_image)
    
    def display_image(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (400, 300))
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        self.image_label.config(image=img_tk, text="")
        self.image_label.image = img_tk
    
    def apply_grayscale(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        self.current_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.display_image(cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2RGB))
    
    def apply_binary(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, self.current_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        self.display_image(cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2RGB))
    
    def apply_median_blur(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.current_image = cv2.medianBlur(gray, 9)
        self.display_image(cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2RGB))
    
    def apply_laplacian(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.current_image = cv2.Laplacian(gray, cv2.CV_64F)
        self.display_image(cv2.cvtColor(np.uint8(np.absolute(self.current_image)), cv2.COLOR_GRAY2RGB))
    
    # ==================== TAB 2: AUDIO PROCESSING ====================
    def create_audio_processing_tab(self):
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Audio Processing (Lab 3)")
        
        # Control Panel
        control_frame = ttk.LabelFrame(frame, text="Audio Processing Controls", padding="10")
        control_frame.pack(fill='x', pady=10)
        
        ttk.Button(control_frame, text="Load Audio File", command=self.load_audio).pack(side='left', padx=5)
        
        filter_frame = ttk.Frame(control_frame)
        filter_frame.pack(side='left', padx=20)
        
        ttk.Label(filter_frame, text="Apply Filter:").pack(side='left', padx=5)
        ttk.Button(filter_frame, text="Lowpass (1kHz)", command=lambda: self.apply_audio_filter('lowpass')).pack(side='left', padx=3)
        ttk.Button(filter_frame, text="Highpass (2kHz)", command=lambda: self.apply_audio_filter('highpass')).pack(side='left', padx=3)
        ttk.Button(filter_frame, text="Bandpass (1-3kHz)", command=lambda: self.apply_audio_filter('bandpass')).pack(side='left', padx=3)
        ttk.Button(filter_frame, text="Bandstop (1-3kHz)", command=lambda: self.apply_audio_filter('bandstop')).pack(side='left', padx=3)
        
        # Matplotlib canvas for audio visualization
        self.audio_fig = Figure(figsize=(10, 4), dpi=80)
        self.audio_canvas = FigureCanvasTkAgg(self.audio_fig, master=frame)
        self.audio_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.audio_data = None
        self.audio_fs = None
        self.filtered_audio = None
        
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
                messagebox.showinfo("Success", f"Loaded: {os.path.basename(filepath)}\nSample Rate: {self.audio_fs} Hz")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load audio: {str(e)}")
    
    def apply_audio_filter(self, filter_type):
        if self.audio_data is None:
            messagebox.showwarning("Warning", "Please load an audio file first")
            return
        
        try:
            nyq = 0.5 * self.audio_fs
            
            if filter_type == 'lowpass':
                norm = 1000 / nyq
                b, a = butter(5, norm, btype='low')
                title = "Lowpass Filtered (1kHz)"
            elif filter_type == 'highpass':
                norm = 2000 / nyq
                b, a = butter(5, norm, btype='high')
                title = "Highpass Filtered (2kHz)"
            elif filter_type == 'bandpass':
                norm = [1000 / nyq, 3000 / nyq]
                b, a = butter(5, norm, btype='band')
                title = "Bandpass Filtered (1-3kHz)"
            else:  # bandstop
                norm = [1000 / nyq, 3000 / nyq]
                b, a = butter(5, norm, btype='bandstop')
                title = "Bandstop Filtered (1-3kHz)"
            
            self.filtered_audio = lfilter(b, a, self.audio_data)
            self.plot_audio(self.filtered_audio, title)
            
            # Save filtered audio
            output_file = f"{filter_type}_filtered.wav"
            sf.write(output_file, self.filtered_audio, self.audio_fs)
            messagebox.showinfo("Success", f"Filtered audio saved as {output_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Filter error: {str(e)}")
    
    def plot_audio(self, signal, title):
        self.audio_fig.clear()
        ax = self.audio_fig.add_subplot(111)
        ax.plot(signal)
        ax.set_title(title)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        self.audio_fig.tight_layout()
        self.audio_canvas.draw()
    
    # ==================== TAB 3: Z-TRANSFORM ====================
    def create_z_transform_tab(self):
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Z-Transform (Lab 4)")
        
        # Info frame
        info_frame = ttk.LabelFrame(frame, text="Instructions", padding="10")
        info_frame.pack(fill='x', pady=10)
        
        info_text = """Enter the sequence values as space-separated numbers.
Example 1: 5 3 -3 0 4 -2  (represents x(n) from n=-2 to n=3)
Example 2: 1 2 0 3 4 5    (represents x(n) from n=0 onwards, starting at index 0)

The Z-transform formula: X(z) = Σ x(n)·z^(-n)"""
        ttk.Label(info_frame, text=info_text, justify='left').pack(fill='x')
        
        # Input frame
        input_frame = ttk.LabelFrame(frame, text="Z-Transform Calculator", padding="10")
        input_frame.pack(fill='x', pady=10)
        
        ttk.Label(input_frame, text="Enter sequence (space-separated numbers):").pack(side='left', padx=5)
        self.z_input = ttk.Entry(input_frame, width=40)
        self.z_input.pack(side='left', padx=5)
        self.z_input.bind('<Return>', lambda e: self.calculate_z_transform())
        
        ttk.Button(input_frame, text="Calculate Z-Transform", command=self.calculate_z_transform).pack(side='left', padx=5)
        
        # Output frame
        output_frame = ttk.LabelFrame(frame, text="Result", padding="10")
        output_frame.pack(fill='both', expand=True, pady=10)
        
        self.z_output = tk.Text(output_frame, height=15, width=80, font=("Courier", 12))
        self.z_output.pack(fill='both', expand=True)
    
    def calculate_z_transform(self):
        try:
            input_str = self.z_input.get().strip()
            if not input_str:
                messagebox.showwarning("Warning", "Please enter a sequence")
                return
            
            sequence = [int(x) for x in input_str.split()]
            
            # Calculate starting index (n_start)
            # If we have 6 elements, and typical examples show x(-2) to x(3):
            # We'll use the length to determine: for 6 elements, start at -(6//2) = -3
            # But more accurately, we start from n=0 by default
            n_start = 0
            
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
        """
        Calculate Z-transform with detailed step-by-step calculation
        X(z) = Σ x(n)·z^(-n)
        """
        terms = []
        equation_steps = []
        
        for idx, val in enumerate(sequence):
            n = n_start + idx
            if val == 0:
                continue
            
            # Create z power term
            if n == 0:
                z_power = "z^0"
                z_power_simplified = "1"
            elif n > 0:
                z_power = f"z^{n}"
                z_power_simplified = f"z^{n}"
            else:  # n < 0
                z_power = f"z^({n})"
                z_power_simplified = f"z^{n}"
            
            # Create the term
            if val == 1:
                term = z_power_simplified
            elif val == -1:
                term = f"-{z_power_simplified}"
            else:
                term = f"{val}{z_power_simplified}"
            
            terms.append(term)
            equation_steps.append(f"  + x({n})·{z_power} = {val}·{z_power_simplified}")
        
        # Join with proper formatting
        if not terms:
            return "0", "0"
        
        result = " + ".join(terms).replace("+ -", "- ")
        detailed = "\n".join(equation_steps)
        
        return result, detailed
    
    # ==================== TAB 4: FFT ANALYSIS ====================
    def create_fft_tab(self):
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="FFT Analysis (Lab 6)")
        
        # Control Panel
        control_frame = ttk.LabelFrame(frame, text="FFT Controls", padding="10")
        control_frame.pack(fill='x', pady=10)
        
        ttk.Label(control_frame, text="Signal Example:").pack(side='left', padx=5)
        ttk.Button(control_frame, text="Generate Signal", command=self.generate_fft_signal).pack(side='left', padx=5)
        
        # Matplotlib canvas
        self.fft_fig = Figure(figsize=(10, 5), dpi=80)
        self.fft_canvas = FigureCanvasTkAgg(self.fft_fig, master=frame)
        self.fft_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def generate_fft_signal(self):
        x = np.array([-1, 5, 7, 2])
        N = len(x)
        
        X = np.fft.fft(x)
        freq = np.fft.fftfreq(N)
        magnitude = np.abs(X)
        
        self.fft_fig.clear()
        
        # Plot 1: Input Signal
        ax1 = self.fft_fig.add_subplot(121)
        ax1.stem(range(N), x, basefmt=' ')
        ax1.set_title("Input Signal x[n]")
        ax1.set_xlabel("n")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Magnitude Spectrum
        ax2 = self.fft_fig.add_subplot(122)
        ax2.stem(freq, magnitude, basefmt=' ')
        ax2.set_title("Magnitude Spectrum |X[k]|")
        ax2.set_xlabel("Frequency")
        ax2.set_ylabel("Magnitude")
        ax2.grid(True, alpha=0.3)
        
        self.fft_fig.tight_layout()
        self.fft_canvas.draw()
        
        # Print results
        print(f"Input Signal: {x}")
        print(f"DFT: {X}")
        print(f"Magnitude: {magnitude}")
    
    # ==================== TAB 5: WINDOWING ====================
    def create_windowing_tab(self):
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Windowing (Lab 7)")
        
        # Control Panel - Left side with parameters
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
        
        # Window selection buttons
        ttk.Label(left_frame, text="Select Window:").pack(anchor='w', pady=(20, 5))
        ttk.Button(left_frame, text="Rectangular", command=lambda: self.apply_window('Rectangular')).pack(fill='x', pady=3)
        ttk.Button(left_frame, text="Blackman", command=lambda: self.apply_window('Blackman')).pack(fill='x', pady=3)
        ttk.Button(left_frame, text="Hann", command=lambda: self.apply_window('Hann')).pack(fill='x', pady=3)
        ttk.Button(left_frame, text="Hamming", command=lambda: self.apply_window('Hamming')).pack(fill='x', pady=3)
        
        # Right side - Display
        right_frame = ttk.Frame(frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        # Matplotlib canvas
        self.window_fig = Figure(figsize=(14, 5), dpi=80)
        self.window_canvas = FigureCanvasTkAgg(self.window_fig, master=right_frame)
        self.window_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Initial plot
        self.apply_window('Rectangular')
    
    def apply_window(self, window_name):
        try:
            N = int(self.window_N.get())
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
        
        X = np.fft.fft(x)
        Xw = np.fft.fft(xw)
        freq = np.arange(N)
        
        self.window_fig.clear()
        
        # Column 1: Original Signal
        ax1 = self.window_fig.add_subplot(131)
        ax1.plot(n, x, 'b-', linewidth=2)
        ax1.set_title("Original Signal")
        ax1.set_xlabel("n")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)
        
        # Column 2: Windowed Signal
        ax2 = self.window_fig.add_subplot(132)
        ax2.plot(n, w, 'r--', linewidth=2, label='Window')
        ax2.plot(n, xw, 'g-', linewidth=2, label='Windowed Signal')
        ax2.set_title(f"{window_name} Window (Time Domain)")
        ax2.set_xlabel("n")
        ax2.set_ylabel("Amplitude")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Column 3: FFT Comparison
        ax3 = self.window_fig.add_subplot(133)
        ax3.plot(freq, np.abs(X), 'b-', label="Original", linewidth=2)
        ax3.plot(freq, np.abs(Xw), 'r-', label="Windowed", linewidth=2)
        ax3.set_title("FFT Comparison")
        ax3.set_xlabel("Frequency Bin")
        ax3.set_ylabel("Magnitude")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        self.window_fig.suptitle(f"{window_name} Window Analysis (N={N}, f1={f1}, f2={f2}, A2={A2})", 
                                fontsize=12, fontweight='bold')
        self.window_fig.tight_layout()
        self.window_canvas.draw()
    
    # ==================== TAB 6: SAMPLING ====================
    def create_sampling_tab(self):
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Sampling Theory (Labs 1,5)")
        
        # Control Panel
        control_frame = ttk.LabelFrame(frame, text="Sampling Examples", padding="10")
        control_frame.pack(fill='x', pady=10)
        
        ttk.Button(control_frame, text="Continuous Signal (1kHz)", command=self.plot_first_task).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Sampling 1kHz @ 8kHz", command=self.plot_second_task).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Sampling 2kHz @ 8kHz", command=self.plot_third_task).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Sampling 6kHz @ 8kHz (Aliasing)", command=self.plot_fourth_task).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Multiple Frequencies", command=self.plot_fifth_task).pack(side='left', padx=5)
        
        # Matplotlib canvas
        self.sampling_fig = Figure(figsize=(10, 5), dpi=80)
        self.sampling_canvas = FigureCanvasTkAgg(self.sampling_fig, master=frame)
        self.sampling_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def plot_first_task(self):
        t_min, t_max, num_t = -1, 1, 1000
        t = np.linspace(t_min, t_max, num_t)
        xt = np.cos(2 * np.pi * 1000 * t)
        
        self.sampling_fig.clear()
        ax = self.sampling_fig.add_subplot(111)
        ax.plot(t, xt)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("First Task: Continuous 1kHz Cosine Signal")
        ax.grid(True, alpha=0.3)
        self.sampling_fig.tight_layout()
        self.sampling_canvas.draw()
    
    def plot_second_task(self):
        f, fs = 1000, 8000
        t = np.linspace(0, 0.01, 10000)
        ts = np.arange(0, 0.01, 1/fs)
        
        x_t = np.cos(2 * np.pi * f * t)
        x_sample = np.cos(2 * np.pi * f * ts)
        
        self.sampling_fig.clear()
        ax = self.sampling_fig.add_subplot(111)
        ax.plot(t, x_t, label="Continuous Signal", linewidth=2)
        ax.stem(ts, x_sample, linefmt='r-', markerfmt='ro', basefmt="", label="Sampled (fs=8kHz)")
        ax.set_title("Sampling 1kHz at 8kHz (Nyquist Satisfied)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.sampling_fig.tight_layout()
        self.sampling_canvas.draw()
    
    def plot_third_task(self):
        f, fs = 2000, 8000
        t = np.linspace(0, 0.01, 10000)
        ts = np.arange(0, 0.01, 1/fs)
        
        x_t = np.cos(2 * np.pi * f * t)
        x_sample = np.cos(2 * np.pi * f * ts)
        
        self.sampling_fig.clear()
        ax = self.sampling_fig.add_subplot(111)
        ax.plot(t, x_t, label="Continuous Signal", linewidth=2)
        ax.stem(ts, x_sample, linefmt='r-', markerfmt='ro', basefmt="", label="Sampled (fs=8kHz)")
        ax.set_title("Sampling 2kHz at 8kHz (Nyquist Satisfied)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.sampling_fig.tight_layout()
        self.sampling_canvas.draw()
    
    def plot_fourth_task(self):
        f, fs = 6000, 8000
        t = np.linspace(0, 0.01, 10000)
        ts = np.arange(0, 0.01, 1/fs)
        
        x_t = np.cos(2 * np.pi * f * t)
        x_sample = np.cos(2 * np.pi * f * ts)
        
        self.sampling_fig.clear()
        ax = self.sampling_fig.add_subplot(111)
        ax.plot(t, x_t, label="Continuous Signal (6kHz)", linewidth=2)
        ax.stem(ts, x_sample, linefmt='r-', markerfmt='ro', basefmt="", label="Sampled (fs=8kHz)")
        ax.set_title("⚠️ Sampling 6kHz at 8kHz (ALIASING - Nyquist Violated!)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.sampling_fig.tight_layout()
        self.sampling_canvas.draw()
    
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
        ax.set_xlabel("n (samples)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.sampling_fig.tight_layout()
        self.sampling_canvas.draw()

def main():
    root = tk.Tk()
    app = DSPMasterProgram(root)
    root.mainloop()

if __name__ == "__main__":
    main()
