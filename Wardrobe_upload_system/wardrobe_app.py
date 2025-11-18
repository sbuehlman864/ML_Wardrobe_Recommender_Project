"""
Wardrobe Upload System
A GUI application that captures camera feed, detects clothing items using YOLO,
and stores them in person-specific folders.
"""

import cv2
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import os
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import torch

# Mapping of YOLO class IDs to Fashion Dataset articleType categories
# This mapping should be adjusted based on the YOLO model used
# For a fashion-specific model, map model classes to articleType values

# Common articleType values from Fashion Product Images Dataset
ARTICLE_TYPES = [
    'Shirts', 'Tshirts', 'Jeans', 'Track Pants', 'Shorts', 'Tops',
    'Casual Shoes', 'Formal Shoes', 'Sports Shoes', 'Sandals', 'Flip Flops', 'Flats', 'Heels',
    'Watches', 'Belts', 'Handbags', 'Socks', 'Bra', 'Briefs',
    'Sweatshirts', 'Kurtas', 'Waistcoat', 'Sarees', 'Sunglasses',
    'Bracelet', 'Pendant', 'Scarves', 'Laptop Bag', 'Shoe Accessories', 'Innerwear Vests'
]

# Class mapping for kesimeg/yolov8n-clothing-detection model
# This model detects 4 categories: Clothing, Shoes, Bags, Accessories
# Source: https://huggingface.co/kesimeg/yolov8n-clothing-detection
CLOTHING_CLASS_MAPPING = {
    # Direct mappings from model classes to articleType
    'clothing': 'Tshirts',  # Default to Tshirts for clothing category
    'shoes': 'Casual Shoes',
    'bags': 'Handbags',
    'accessories': 'Watches',  # Default to Watches for accessories
    
    # Lowercase variants
    'shoe': 'Casual Shoes',
    'bag': 'Handbags',
    'accessory': 'Watches',
}

# For standard YOLOv8, we'll use a different approach
# We can use a fashion-specific model or implement custom detection
# Option: Use YOLOv8-seg for segmentation or a fashion-specific model


class WardrobeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wardrobe Upload System")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.camera = None
        self.is_capturing = False
        self.current_person = None
        self.person_folder = None
        self.detected_items = {}  # Track detected items to avoid duplicates
        self.save_counter = 0
        
        # Load YOLO model
        # Try to load pre-trained clothing detection models
        self.model = None
        self.model_type = None
        
        # Try pre-trained clothing detection model paths
        fashion_model_paths = [
            'yolov8n-clothing-detection.pt',  # Standard name
            'best.pt',  # Common Hugging Face model name
            'fashion_yolo.pt',  # Custom fashion model
            'yolov8n-fashion.pt',  # Alternative name
            'yolov8-fashion.pt',  # Another alternative
            'fashion_model.pt',  # Generic name
        ]
        
        # Try loading from local files first
        for model_path in fashion_model_paths:
            try:
                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                    self.model_type = 'fashion'
                    print(f"✓ Fashion YOLO model loaded: {model_path}")
                    break
            except Exception as e:
                print(f"Error loading {model_path}: {e}")
                continue
        
        # If not found locally, try downloading from Hugging Face
        if self.model is None:
            print("No local model found. Attempting to load from Hugging Face...")
            try:
                # Method 1: Try ultralyticsplus (if available)
                try:
                    from ultralyticsplus import YOLO as YOLOPlus
                    print("Loading pre-trained model from Hugging Face (ultralyticsplus)...")
                    hf_model = YOLOPlus('kesimeg/yolov8n-clothing-detection')
                    self.model = hf_model
                    self.model_type = 'fashion'
                    print("✓ Loaded pre-trained clothing detection model from Hugging Face")
                    return  # Success, exit early
                except ImportError:
                    print("ultralyticsplus not available, trying alternative method...")
                
                # Method 2: Try huggingface_hub (lighter dependency)
                try:
                    from huggingface_hub import hf_hub_download
                    print("Downloading model using huggingface_hub...")
                    
                    # Try different possible filenames
                    possible_filenames = [
                        "best.pt",  # Most common name
                        "yolov8n-clothing-detection.pt",
                        "model.pt",
                        "weights.pt"
                    ]
                    
                    model_path = None
                    for filename in possible_filenames:
                        try:
                            model_path = hf_hub_download(
                                repo_id="kesimeg/yolov8n-clothing-detection",
                                filename=filename,
                                local_dir="./",
                                local_dir_use_symlinks=False
                            )
                            if os.path.exists(model_path):
                                print(f"✓ Found model file: {filename}")
                                break
                        except Exception as e:
                            continue
                    
                    if model_path and os.path.exists(model_path):
                        # Rename to standard name for consistency
                        standard_name = "yolov8n-clothing-detection.pt"
                        if model_path != standard_name and not os.path.exists(standard_name):
                            try:
                                import shutil
                                shutil.copy2(model_path, standard_name)
                                model_path = standard_name
                                print(f"✓ Saved as: {standard_name}")
                            except:
                                pass  # Use original name if rename fails
                        
                        self.model = YOLO(model_path)
                        self.model_type = 'fashion'
                        print(f"✓ Loaded model from: {model_path}")
                        return  # Success
                except ImportError:
                    print("huggingface_hub not installed, trying to install...")
                    import subprocess
                    import sys
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "--quiet"])
                        from huggingface_hub import hf_hub_download
                        
                        # Try different filenames
                        possible_filenames = ["best.pt", "yolov8n-clothing-detection.pt", "model.pt", "weights.pt"]
                        model_path = None
                        for filename in possible_filenames:
                            try:
                                model_path = hf_hub_download(
                                    repo_id="kesimeg/yolov8n-clothing-detection",
                                    filename=filename,
                                    local_dir="./"
                                )
                                if os.path.exists(model_path):
                                    break
                            except:
                                continue
                        
                        if model_path and os.path.exists(model_path):
                            self.model = YOLO(model_path)
                            self.model_type = 'fashion'
                            print(f"✓ Loaded model from: {model_path}")
                            return  # Success
                    except Exception as install_error:
                        print(f"Could not install huggingface_hub: {install_error}")
                except Exception as hf_error:
                    print(f"Could not download from Hugging Face: {hf_error}")
                
                # Method 3: Try alternative loader script
                try:
                    from load_model_alternative import load_model_alternative
                    print("Trying alternative model loader...")
                    alt_model = load_model_alternative()
                    if alt_model:
                        self.model = alt_model
                        self.model_type = 'fashion'
                        print("✓ Loaded model using alternative method")
                        return  # Success
                except Exception as alt_error:
                    print(f"Alternative loader failed: {alt_error}")
                    
            except Exception as e:
                print(f"All download methods failed: {e}")
            
            # If all methods failed, show helpful error
            if self.model is None:
                error_msg = (
                    "Pre-trained clothing detection model not found!\n\n"
                    "Download options:\n\n"
                    "Option 1 (Easiest):\n"
                    "  pip install huggingface_hub\n"
                    "  (Then restart the app - it will auto-download)\n\n"
                    "Option 2:\n"
                    "  python load_model_alternative.py\n\n"
                    "Option 3:\n"
                    "  Visit: https://huggingface.co/kesimeg/yolov8n-clothing-detection\n"
                    "  Download the .pt file and save as: yolov8n-clothing-detection.pt\n\n"
                    "Option 4:\n"
                    "  Use standard YOLO (less accurate but works)\n"
                    "  The app will detect person and extract regions"
                )
                print(f"\n{error_msg}")
                messagebox.showwarning("Model Not Found", error_msg)
                self.model = None
        
        # Create storage directory
        self.storage_dir = "wardrobe_storage"
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Create GUI
        self.create_gui()
        
    def create_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Person name input
        ttk.Label(control_frame, text="Person Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.person_name_var = tk.StringVar()
        person_entry = ttk.Entry(control_frame, textvariable=self.person_name_var, width=20)
        person_entry.grid(row=0, column=1, pady=5, padx=5)
        
        ttk.Button(control_frame, text="Set Person", 
                  command=self.set_person).grid(row=1, column=0, columnspan=2, pady=5, sticky=tk.W+tk.E)
        
        # Current person display
        self.current_person_label = ttk.Label(control_frame, text="No person selected", 
                                             foreground="red")
        self.current_person_label.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(row=3, column=0, columnspan=2, 
                                                              sticky=tk.W+tk.E, pady=10)
        
        # Camera controls
        ttk.Button(control_frame, text="Start Camera", 
                  command=self.start_camera).grid(row=4, column=0, columnspan=2, pady=5, sticky=tk.W+tk.E)
        ttk.Button(control_frame, text="Stop Camera", 
                  command=self.stop_camera).grid(row=5, column=0, columnspan=2, pady=5, sticky=tk.W+tk.E)
        
        # Capture controls
        ttk.Separator(control_frame, orient='horizontal').grid(row=6, column=0, columnspan=2, 
                                                              sticky=tk.W+tk.E, pady=10)
        
        ttk.Button(control_frame, text="Capture & Save Clothes", 
                  command=self.capture_clothes).grid(row=7, column=0, columnspan=2, pady=5, sticky=tk.W+tk.E)
        
        # Statistics
        self.stats_label = ttk.Label(control_frame, text="Items captured: 0")
        self.stats_label.grid(row=8, column=0, columnspan=2, pady=10)
        
        # View wardrobe button
        ttk.Separator(control_frame, orient='horizontal').grid(row=9, column=0, columnspan=2, 
                                                              sticky=tk.W+tk.E, pady=10)
        ttk.Button(control_frame, text="View Wardrobe", 
                  command=self.view_wardrobe).grid(row=10, column=0, columnspan=2, pady=5, sticky=tk.W+tk.E)
        
        # Right panel - Camera feed
        camera_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="10")
        camera_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        camera_frame.columnconfigure(0, weight=1)
        camera_frame.rowconfigure(0, weight=1)
        
        # Video display
        self.video_label = ttk.Label(camera_frame, text="Camera not started", 
                                     background="black", foreground="white")
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X)
        
    def set_person(self):
        """Set the current person name and create folder"""
        name = self.person_name_var.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter a person name")
            return
        
        self.current_person = name
        self.person_folder = os.path.join(self.storage_dir, name)
        os.makedirs(self.person_folder, exist_ok=True)
        
        self.current_person_label.config(text=f"Current: {name}", foreground="green")
        self.detected_items = {}  # Reset for new person
        self.save_counter = 0
        self.update_stats()
        self.update_status(f"Person set to: {name}")
        
    def start_camera(self):
        """Start camera feed"""
        if not self.current_person:
            messagebox.showwarning("Warning", "Please set a person name first")
            return
        
        if not self.model:
            messagebox.showerror("Error", 
                "Fashion YOLO model not loaded!\n\n"
                "Please train a fashion-specific YOLO model and place it in this directory.\n\n"
                "See TRAINING_FASHION_YOLO.md for instructions.")
            return
        
        if self.is_capturing:
            return
        
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            self.is_capturing = True
            self.update_status("Camera started - Detecting clothing items directly")
            self.update_camera_feed()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
            self.is_capturing = False
    
    def stop_camera(self):
        """Stop camera feed"""
        self.is_capturing = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.video_label.config(image='', text="Camera stopped")
        self.update_status("Camera stopped")
    
    def update_camera_feed(self):
        """Update camera feed display"""
        if not self.is_capturing or not self.camera:
            return
        
        ret, frame = self.camera.read()
        if not ret:
            self.update_status("Failed to read from camera")
            return
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect clothing/persons using YOLO
        if self.model:
            annotated_frame = self.detect_clothing(frame)
        else:
            annotated_frame = frame
        
        # Convert to RGB for tkinter
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Resize to fit display
        display_width = 800
        display_height = 600
        frame_pil = frame_pil.resize((display_width, display_height), Image.Resampling.LANCZOS)
        
        frame_tk = ImageTk.PhotoImage(image=frame_pil)
        self.video_label.config(image=frame_tk, text="")
        self.video_label.image = frame_tk  # Keep a reference
        
        # Schedule next update
        self.root.after(30, self.update_camera_feed)  # ~30 FPS
    
    def detect_clothing(self, frame):
        """
        Detect clothing items in frame using YOLO.
        
        The kesimeg/yolov8n-clothing-detection model detects:
        - Clothing (class 0)
        - Shoes (class 1)
        - Bags (class 2)
        - Accessories (class 3)
        """
        if not self.model:
            return frame
        
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        # Draw bounding boxes
        annotated_frame = frame.copy()
        detections = []
        
        # Get class names from model
        class_names = self.model.names if hasattr(self.model, 'names') else {}
        
        # Debug: Print model class names on first detection
        if not hasattr(self, '_class_names_printed'):
            print(f"Model class names: {class_names}")
            self._class_names_printed = True
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Get class name - handle both dict and list formats
                if isinstance(class_names, dict):
                    class_name = class_names.get(cls_id, f"class_{cls_id}")
                elif isinstance(class_names, list) and cls_id < len(class_names):
                    class_name = class_names[cls_id]
                else:
                    class_name = f"class_{cls_id}"
                
                class_name_lower = str(class_name).lower()
                article_type = self.map_to_article_type(class_name_lower, cls_id)
                
                # Process all detections from the 4-category model
                # Lower confidence threshold (0.3) since this is a specialized model
                if conf > 0.3:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    color = self.get_color_for_article_type(article_type)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Show both original class name and mapped article type
                    label = f"{article_type} ({class_name}) {conf:.2f}"
                    cv2.putText(annotated_frame, label, 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class': class_name,
                        'class_id': cls_id,
                        'article_type': article_type
                    })
        
        # Store current detections for capture
        self.current_detections = detections
        
        return annotated_frame
    
    def map_to_article_type(self, class_name, cls_id):
        """
        Map YOLO class name/ID to Fashion Dataset articleType.
        
        The kesimeg/yolov8n-clothing-detection model detects 4 categories:
        - Clothing
        - Shoes
        - Bags
        - Accessories
        
        Source: https://huggingface.co/kesimeg/yolov8n-clothing-detection
        """
        class_name_lower = class_name.lower().strip()
        
        # Check custom mapping first
        if class_name_lower in CLOTHING_CLASS_MAPPING:
            return CLOTHING_CLASS_MAPPING[class_name_lower]
        
        # Direct mapping for the 4 categories from the model
        category_mapping = {
            'clothing': 'Tshirts',  # Default clothing type
            'shoes': 'Casual Shoes',
            'shoe': 'Casual Shoes',
            'bags': 'Handbags',
            'bag': 'Handbags',
            'accessories': 'Watches',  # Default accessory type
            'accessory': 'Watches',
        }
        
        # Check exact match
        if class_name_lower in category_mapping:
            return category_mapping[class_name_lower]
        
        # Check partial matches for the 4 main categories
        if 'cloth' in class_name_lower:
            return 'Tshirts'  # Default to Tshirts for clothing
        elif 'shoe' in class_name_lower or 'footwear' in class_name_lower:
            return 'Casual Shoes'
        elif 'bag' in class_name_lower:
            return 'Handbags'
        elif 'accessor' in class_name_lower:
            return 'Watches'  # Default to Watches for accessories
        
        # Fallback: Try to map based on class ID if available
        # The model has 4 classes: 0=Clothing, 1=Shoes, 2=Bags, 3=Accessories
        if cls_id is not None:
            id_mapping = {
                0: 'Tshirts',  # Clothing
                1: 'Casual Shoes',  # Shoes
                2: 'Handbags',  # Bags
                3: 'Watches',  # Accessories
            }
            if cls_id in id_mapping:
                return id_mapping[cls_id]
        
        # Default fallback
        return 'Clothing'
    
    def get_color_for_article_type(self, article_type):
        """Get color for bounding box based on article type"""
        # Color mapping for different article types
        color_map = {
            'Shirts': (0, 255, 0),      # Green
            'Tshirts': (0, 255, 255),   # Yellow
            'Jeans': (255, 0, 0),       # Blue
            'Casual Shoes': (255, 0, 255),  # Magenta
            'Sports Shoes': (128, 0, 128),  # Purple
            'Watches': (0, 165, 255),   # Orange
            'Handbags': (255, 192, 203),  # Pink
            'Belts': (255, 255, 0),     # Cyan
            'Socks': (128, 128, 128),   # Gray
            'Shorts': (0, 128, 255),    # Light Blue
            'Tops': (255, 20, 147),     # Deep Pink
        }
        return color_map.get(article_type, (0, 255, 0))  # Default green
    
    
    def capture_clothes(self):
        """Capture and save detected clothing items"""
        if not self.is_capturing or not self.camera:
            messagebox.showwarning("Warning", "Please start camera first")
            return
        
        if not self.current_person:
            messagebox.showwarning("Warning", "Please set a person name first")
            return
        
        if not hasattr(self, 'current_detections') or not self.current_detections:
            messagebox.showinfo("Info", 
                "No clothing items detected.\n\n"
                "Please ensure clothing items are clearly visible in the camera frame.\n"
                "The system directly detects specific clothing items (Shirts, Jeans, Shoes, etc.) "
                "using the fashion YOLO model.")
            return
        
        # Capture current frame
        ret, frame = self.camera.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture frame")
            return
        
        frame = cv2.flip(frame, 1)  # Flip for consistency
        
        saved_count = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, detection in enumerate(self.current_detections):
            x1, y1, x2, y2 = detection['bbox']
            
            # Crop detected region with padding
            padding = 10  # Add padding around bounding box
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(frame.shape[1], x2 + padding)
            y2_pad = min(frame.shape[0], y2 + padding)
            
            cropped = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if cropped.size == 0:
                continue
            
            # Get article type for filename
            article_type = detection.get('article_type', 'Clothing')
            # Clean article type for filename (remove spaces, special chars)
            article_type_clean = article_type.replace(' ', '_').replace('/', '_')
            
            # Create unique filename with article type
            filename = f"{article_type_clean}_{timestamp}_{self.save_counter:04d}.jpg"
            filepath = os.path.join(self.person_folder, filename)
            
            # Save cropped image
            cv2.imwrite(filepath, cropped)
            saved_count += 1
            self.save_counter += 1
            
            # Optionally save metadata (article type, confidence, etc.)
            # Could save to a JSON file for later use
        
        if saved_count > 0:
            self.update_stats()
            self.update_status(f"Saved {saved_count} clothing item(s)")
            messagebox.showinfo("Success", f"Saved {saved_count} clothing item(s) to {self.current_person}'s wardrobe")
        else:
            messagebox.showwarning("Warning", "No items were saved")
    
    def update_stats(self):
        """Update statistics display"""
        if self.person_folder and os.path.exists(self.person_folder):
            count = len([f for f in os.listdir(self.person_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
            self.stats_label.config(text=f"Items captured: {count}")
        else:
            self.stats_label.config(text="Items captured: 0")
    
    def view_wardrobe(self):
        """Open wardrobe viewer window"""
        if not self.current_person:
            # Show person selection dialog
            persons = [d for d in os.listdir(self.storage_dir) 
                      if os.path.isdir(os.path.join(self.storage_dir, d))]
            
            if not persons:
                messagebox.showinfo("Info", "No wardrobes found. Please capture some items first.")
                return
            
            # Create selection dialog
            person = simpledialog.askstring("Select Person", 
                                          f"Enter person name:\nAvailable: {', '.join(persons)}")
            if not person:
                return
            self.current_person = person
        
        # Open wardrobe viewer
        WardrobeViewer(self.root, self.storage_dir, self.current_person)
    
    def update_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)
    
    def __del__(self):
        """Cleanup on exit"""
        self.stop_camera()


class WardrobeViewer:
    """Window to view stored wardrobe items for a person"""
    
    def __init__(self, parent, storage_dir, person_name):
        self.storage_dir = storage_dir
        self.person_name = person_name
        self.person_folder = os.path.join(storage_dir, person_name)
        
        # Create window
        self.window = tk.Toplevel(parent)
        self.window.title(f"Wardrobe - {person_name}")
        self.window.geometry("1000x700")
        
        # Get all images
        self.images = [f for f in os.listdir(self.person_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not self.images:
            ttk.Label(self.window, text="No items found in wardrobe", 
                     font=("Arial", 14)).pack(pady=50)
            return
        
        # Create scrollable frame
        canvas = tk.Canvas(self.window)
        scrollbar = ttk.Scrollbar(self.window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Display images in grid
        cols = 3
        for idx, img_name in enumerate(self.images):
            row = idx // cols
            col = idx % cols
            
            img_path = os.path.join(self.person_folder, img_name)
            
            # Load and resize image
            try:
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                
                # Resize for display
                img_pil.thumbnail((250, 250), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(img_pil)
                
                # Create frame for image
                img_frame = ttk.Frame(scrollable_frame)
                img_frame.grid(row=row, column=col, padx=10, pady=10)
                
                # Image label
                img_label = ttk.Label(img_frame, image=img_tk)
                img_label.image = img_tk  # Keep reference
                img_label.pack()
                
                # Filename label
                ttk.Label(img_frame, text=img_name, font=("Arial", 8)).pack()
                
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Info label
        info_label = ttk.Label(self.window, 
                              text=f"Total items: {len(self.images)}", 
                              font=("Arial", 10))
        info_label.pack(pady=10)


def main():
    root = tk.Tk()
    app = WardrobeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

