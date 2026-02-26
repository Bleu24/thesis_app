import customtkinter as ctk
from tkinter import filedialog, messagebox
import torch
import torch.nn.functional as F
import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import logging
import sys
import traceback

from models import UNet, ResNetRegression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ThesisApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Rice Yield System (Interactive MVP)")
        self.geometry("1300x850")
        sys.excepthook = self.handle_crash

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Nueva Ecija\nYield System", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.lbl_controls = ctk.CTkLabel(self.sidebar_frame, text="--- Model Settings ---", text_color="gray")
        self.lbl_controls.grid(row=1, column=0, pady=(10, 0))

        self.lbl_scale = ctk.CTkLabel(self.sidebar_frame, text="Data Divisor:")
        self.lbl_scale.grid(row=2, column=0, padx=20, sticky="w")
        self.scale_var = ctk.StringVar(value="1000.0")
        self.opt_scale = ctk.CTkOptionMenu(self.sidebar_frame, values=["1000.0", "10000.0", "1.0"], variable=self.scale_var)
        self.opt_scale.grid(row=3, column=0, padx=20, pady=(0, 10))

        self.lbl_thresh = ctk.CTkLabel(self.sidebar_frame, text="U-Net Threshold: 0.50")
        self.lbl_thresh.grid(row=4, column=0, padx=20, sticky="w")
        
        self.slider_thresh = ctk.CTkSlider(self.sidebar_frame, from_=0.0, to=1.0, number_of_steps=100, command=self.update_thresh_lbl)
        self.slider_thresh.set(0.50)
        self.slider_thresh.grid(row=5, column=0, padx=20, pady=(0, 10))

        self.ndvi_var = ctk.BooleanVar(value=True)
        self.chk_ndvi = ctk.CTkCheckBox(self.sidebar_frame, text="Use NDVI Filter", variable=self.ndvi_var)
        self.chk_ndvi.grid(row=6, column=0, padx=20, pady=(0, 20))

        self.btn_load = ctk.CTkButton(self.sidebar_frame, text="1. Load Sentinel-2", command=self.load_image)
        self.btn_load.grid(row=7, column=0, padx=20, pady=5)

        self.btn_load_csv = ctk.CTkButton(self.sidebar_frame, text="2. Load Features (.csv)", command=self.load_csv)
        self.btn_load_csv.grid(row=8, column=0, padx=20, pady=5)

        self.csv_status = ctk.CTkLabel(self.sidebar_frame, text="CSV: None", text_color="gray")
        self.csv_status.grid(row=9, column=0, padx=20, pady=(0, 10))

        self.btn_segment = ctk.CTkButton(self.sidebar_frame, text="3. Run Segmentation", command=self.start_segmentation_thread, state="disabled")
        self.btn_segment.grid(row=10, column=0, padx=20, pady=5)

        self.btn_estimate = ctk.CTkButton(self.sidebar_frame, text="4. Estimate Yield", command=self.start_estimation_thread, state="disabled")
        self.btn_estimate.grid(row=11, column=0, padx=20, pady=5)

        self.btn_clear = ctk.CTkButton(self.sidebar_frame, text="Clear / Reset", command=self.reset_app, fg_color="#C62828", hover_color="#B71C1C")
        self.btn_clear.grid(row=12, column=0, padx=20, pady=(20, 10))

        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Status: Ready", text_color="gray", wraplength=180)
        self.status_label.grid(row=13, column=0, padx=20, pady=10)

        self.yield_label = ctk.CTkLabel(self.sidebar_frame, text="Predicted Yield:", font=ctk.CTkFont(size=14))
        self.yield_label.grid(row=14, column=0, padx=20, pady=(10, 0))
        
        self.yield_value = ctk.CTkLabel(self.sidebar_frame, text="-- t/ha", text_color="#2CC985", font=ctk.CTkFont(size=30, weight="bold"))
        self.yield_value.grid(row=15, column=0, padx=20, pady=5)

        # Main Display Area
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.figure, self.ax = plt.subplots(1, 2, figsize=(9, 4.5))
        self.ax[0].set_title("RGB Composite")
        self.ax[1].set_title("Rice Mask (U-Net)")
        self.ax[0].axis('off')
        self.ax[1].axis('off')

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.image_path = None
        self.raw_data = None    
        self.mask_data = None
        self.raw_mask_probs = None 
        self.aux_df = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.load_models()

    def update_thresh_lbl(self, value):
        self.lbl_thresh.configure(text=f"U-Net Threshold: {value:.2f}")
        if self.raw_mask_probs is not None:
            self.apply_threshold()

    def handle_crash(self, exc_type, exc_value, exc_traceback):
        error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logging.critical(f"UNHANDLED EXCEPTION:\n{error_msg}")
        messagebox.showerror("Critical Error", f"App crashed. See log.\n{exc_value}")

    def load_models(self):
        try:
            logging.info("Loading models...")
            self.unet = UNet(in_channels=7, out_channels=1, base_channels=64, max_channels=512, depth=5, norm="group").to(self.device)
            self.resnet = ResNetRegression(in_channels=7, aux_features=14, fusion_hidden=64).to(self.device)

            self.resnet_cfg = {}

            def smart_load(model, path, is_resnet=False):
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                
                if isinstance(checkpoint, dict):
                    if is_resnet and "config" in checkpoint:
                        self.resnet_cfg = checkpoint["config"] 
                    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
                else:
                    state_dict = checkpoint

                new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                
                try:
                    model.load_state_dict(new_state_dict, strict=True)
                    logging.info(f"Loaded {path} (Strict)")
                except RuntimeError:
                    model.load_state_dict(new_state_dict, strict=False)
                    logging.warning(f"Loaded {path} (Non-Strict)")

            smart_load(self.unet, "checkpoints/unet_best.pth", is_resnet=False)
            smart_load(self.resnet, "checkpoints/resnet_best.pth", is_resnet=True)
            self.unet.eval()
            self.resnet.eval()
            logging.info("Models loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            self.status_label.configure(text="Error: Checkpoints Failed")
            self.unet = None

    def reset_app(self):
        self.image_path = None
        self.raw_data = None
        self.mask_data = None
        self.raw_mask_probs = None
        self.aux_df = None
        self.csv_status.configure(text="CSV: None", text_color="gray")
        self.ax[0].clear()
        self.ax[1].clear()
        self.ax[0].set_title("RGB Composite")
        self.ax[1].set_title("Rice Mask (U-Net)")
        self.ax[0].axis('off')
        self.ax[1].axis('off')
        self.canvas.draw()
        self.btn_segment.configure(state="disabled")
        self.btn_estimate.configure(state="disabled")
        self.yield_value.configure(text="-- t/ha")
        self.status_label.configure(text="Status: Cleared")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("GeoTIFF", "*.tif"), ("All Files", "*.*")])
        if file_path:
            self.image_path = file_path
            self.status_label.configure(text="Loading Image...")
            
            try:
                with rasterio.open(file_path) as src:
                    img = src.read() 
                    
                    if img.shape[0] > 7:
                        img = img[:7]
                    elif img.shape[0] < 7:
                        padded = np.zeros((7, img.shape[1], img.shape[2]), dtype=img.dtype)
                        padded[:img.shape[0]] = img
                        img = padded
                        
                    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # 1. SAVE THE RAW, UNSCALED DATA FOR RESNET
                    self.resnet_raw_data = img.astype(np.float32)
                    
                    # 2. CRUSH THE DATA FOR U-NET
                    img_float = img.astype(np.float32)
                    scale_val = float(self.opt_scale.get())
                    if scale_val > 0:
                        img_float *= (1.0 / scale_val)
                    
                    np.clip(img_float, 0.0, 1.0, out=img_float)
                    self.raw_data = img_float
                    
                    if img_float.shape[0] >= 3:
                        rgb = img_float[[2, 1, 0], :, :]
                        rgb = np.transpose(rgb, (1, 2, 0)) 
                        p2, p98 = np.percentile(rgb, (2, 98))
                        rgb_stretched = (rgb - p2) / (p98 - p2)
                        rgb_stretched = np.clip(rgb_stretched, 0, 1)

                        self.ax[0].imshow(rgb_stretched)
                        self.ax[0].set_title("Input (RGB)")
                        self.canvas.draw()
                        self.btn_segment.configure(state="normal")
                        self.status_label.configure(text="Image Loaded")
            except Exception as e:
                logging.error(f"Load Error: {e}")
                self.status_label.configure(text="Error Loading Image")

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if file_path:
            try:
                self.aux_df = pd.read_csv(file_path)
                self.csv_status.configure(text="CSV: Loaded", text_color="#2CC985")
                logging.info(f"Loaded features from {file_path}")
            except Exception as e:
                logging.error(f"CSV Load Error: {e}")
                self.csv_status.configure(text="CSV: Error", text_color="#C62828")

    def start_segmentation_thread(self):
        self.btn_segment.configure(state="disabled")
        self.status_label.configure(text="Running U-Net...")
        threading.Thread(target=self.run_segmentation_task, daemon=True).start()

    def start_estimation_thread(self):
        self.btn_estimate.configure(state="disabled")
        self.status_label.configure(text="Running ResNet...")
        threading.Thread(target=self.run_estimation_task, daemon=True).start()

    def predict_full_image(self, model, image, tile_size=512):
        c, h, w = image.shape
        full_mask_probs = np.zeros((h, w), dtype=np.float32)
        full_tensor = torch.from_numpy(image).float().to(self.device)
        stride = tile_size 

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y2 = min(y + tile_size, h)
                x2 = min(x + tile_size, w)
                
                tile = full_tensor[:, y:y2, x:x2].unsqueeze(0)
                
                pad_h = tile_size - tile.shape[2]
                pad_w = tile_size - tile.shape[3]
                
                h_unpad = tile.shape[2]
                w_unpad = tile.shape[3]

                if pad_h > 0 or pad_w > 0:
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
                
                try:
                    with torch.no_grad():
                        output = model(tile)
                        pred = torch.sigmoid(output)
                        
                        pred = pred[0, 0, :h_unpad, :w_unpad]
                        full_mask_probs[y:y2, x:x2] = pred.cpu().numpy()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                    raise e

        if self.ndvi_var.get():
            red = image[2, :, :]
            nir = image[3, :, :]
            ndvi = (nir - red) / (nir + red + 1e-6)
            ndvi_gate = (ndvi >= 0.25).astype(np.float32)
            full_mask_probs = full_mask_probs * ndvi_gate

        return full_mask_probs

    def run_segmentation_task(self):
        if self.raw_data is not None and self.unet:
            try:
                self.raw_mask_probs = self.predict_full_image(self.unet, self.raw_data)
                self.after(0, self.apply_threshold)
            except Exception as e:
                logging.error(f"Seg Crash: {traceback.format_exc()}")
                self.status_label.configure(text="Seg Error")

    def apply_threshold(self):
        if self.raw_mask_probs is None: return

        thresh = self.slider_thresh.get()
        self.mask_data = (self.raw_mask_probs >= thresh).astype(np.float32)
        
        rice_pixels = np.sum(self.mask_data)
        coverage = (rice_pixels / self.mask_data.size) * 100
        
        self.ax[1].clear()
        self.ax[1].imshow(self.mask_data, cmap='Greens', interpolation='bilinear', alpha=0.8, vmin=0, vmax=1)
        self.ax[1].set_title(f"Rice Mask (Cover: {coverage:.1f}%)")
        self.ax[1].axis('off')
        
        self.canvas.draw()
        self.btn_estimate.configure(state="normal")
        self.btn_segment.configure(state="normal")
        self.status_label.configure(text=f"Masked at >= {thresh:.2f}")

    def run_estimation_task(self):
        # Note: We now check for self.resnet_raw_data
        if self.resnet_raw_data is not None and self.mask_data is not None and self.resnet:
            try:
                # 1. Use the massive, unscaled numbers for ResNet masking!
                masked_img = self.resnet_raw_data * self.mask_data[None, :, :]
                
                mean_y = self.resnet_cfg.get("mean_y", 0.0)
                std_y = self.resnet_cfg.get("std_y", 1.0)
                img_norm = self.resnet_cfg.get("image_norm", {"mean": np.zeros(7), "std": np.ones(7)})
                aux_norm = self.resnet_cfg.get("aux_norm", {"mean": np.zeros(14), "std": np.ones(14)})
                
                # 2. Standardize Image
                img_std_safe = np.where(img_norm["std"] == 0, 1.0, img_norm["std"])
                normed_img = (masked_img - img_norm["mean"][:, None, None]) / img_std_safe[:, None, None]
                
                # 3. Calculate NDVI Features on the unscaled masked image
                rice_pixels = self.mask_data > 0.5
                if rice_pixels.sum() == 0:
                    rice_feats = np.zeros(6, dtype=np.float32)
                else:
                    rice_vals = masked_img[:, rice_pixels]
                    ndvi_early = rice_vals[4]
                    ndvi_late = rice_vals[6]
                    rice_feats = np.array([
                        rice_pixels.mean(),
                        np.mean(ndvi_late),
                        np.std(ndvi_late),
                        np.max(ndvi_late),
                        np.percentile(ndvi_late, 90),
                        np.mean(ndvi_late) - np.mean(ndvi_early)
                    ], dtype=np.float32)
                    
                # 4. Handle Tabular CSV features
                if self.aux_df is not None and "aux_columns" in self.resnet_cfg:
                    cols = self.resnet_cfg["aux_columns"]
                    
                    # Diagnostic check to ensure CSV has what the model expects
                    missing = [c for c in cols if c not in self.aux_df.columns]
                    if missing:
                        logging.warning(f"CSV is missing these columns: {missing}")
                        raise ValueError("Missing columns in CSV")
                        
                    try:
                        csv_feats = self.aux_df.iloc[0][cols].to_numpy(dtype=np.float32)
                        csv_feats = np.nan_to_num(csv_feats, nan=0.0)
                        aux_raw = np.concatenate([csv_feats, rice_feats])
                        logging.info("Successfully bound CSV data and NDVI features.")
                    except Exception as e:
                        logging.warning(f"Error reading CSV rows. Falling back. Error: {e}")
                        aux_raw = np.copy(aux_norm["mean"])
                        aux_raw[-6:] = rice_feats
                else:
                    logging.info("No valid CSV. Falling back to training averages.")
                    aux_raw = np.copy(aux_norm["mean"])
                    aux_raw[-6:] = rice_feats
                
                # 5. Standardize Aux Features
                aux_std_safe = np.where(aux_norm["std"] == 0, 1.0, aux_norm["std"])
                normed_aux = (aux_raw - aux_norm["mean"]) / aux_std_safe
                
                tensor_img = torch.from_numpy(normed_img).float().unsqueeze(0).to(self.device)
                tensor_img = F.interpolate(tensor_img, size=(256, 256), mode='bilinear')
                
                tensor_aux = torch.from_numpy(normed_aux).float().unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.resnet(tensor_img, aux=tensor_aux)
                    z_score = output.item()
                    final_yield = (z_score * std_y) + mean_y
                    logging.info(f"ResNet Z-Score: {z_score:.4f} -> Final Yield: {final_yield:.2f} t/ha")

                self.after(0, lambda: self.update_yield_gui(final_yield))
            except Exception as e:
                logging.error(f"Est Crash: {traceback.format_exc()}")
                self.status_label.configure(text="Est Error")

    def update_yield_gui(self, value):
        self.yield_value.configure(text=f"{value:.2f} t/ha")
        self.btn_estimate.configure(state="normal")
        self.status_label.configure(text="Estimation Done")

if __name__ == "__main__":
    app = ThesisApp()
    app.mainloop()