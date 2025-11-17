"""
Wardrobe Recommender System - Complete GUI Application
Combines wardrobe upload and recommendation system with multi-user support
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from PIL import Image, ImageTk
import cv2
import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add parent directory to path for imports
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(project_root / "Wardrobe_upload_system"))

# Import path utilities for cross-platform compatibility
from path_utils import (
    get_project_root, 
    get_recommender_system_dir,
    get_wardrobe_storage_dir,
    get_product_images_dir,
    normalize_path,
    ensure_dir
)

# Import wardrobe app components
WardrobeApp = None
try:
    # Try importing from Wardrobe_upload_system
    sys.path.insert(0, str(project_root / "Wardrobe_upload_system"))
    from wardrobe_app import WardrobeApp
    print("✓ WardrobeApp imported successfully")
except ImportError as e:
    print(f"Warning: Could not import WardrobeApp: {e}")
    WardrobeApp = None

# Import recommender
Recommender = None
try:
    # Ensure we're importing from the current directory
    import importlib.util
    recommender_path = current_dir / "recommender.py"
    if recommender_path.exists():
        spec = importlib.util.spec_from_file_location("recommender", recommender_path)
        recommender_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recommender_module)
        Recommender = recommender_module.Recommender
        print("✓ Recommender imported successfully")
    else:
        # Try standard import
        from recommender import Recommender
        print("✓ Recommender imported successfully (standard import)")
except ImportError as e:
    print(f"Error: Could not import Recommender: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Looking for: {current_dir / 'recommender.py'}")
    Recommender = None
except Exception as e:
    print(f"Error importing Recommender: {e}")
    import traceback
    traceback.print_exc()
    Recommender = None


class WardrobeUploadTab:
    """Tab for uploading wardrobe items for multiple users"""
    
    def __init__(self, parent, wardrobe_storage_dir=None):
        self.parent = parent
        # Use path_utils to get cross-platform wardrobe storage directory
        if wardrobe_storage_dir is None:
            self.wardrobe_storage_dir = str(get_wardrobe_storage_dir())
        else:
            self.wardrobe_storage_dir = str(normalize_path(wardrobe_storage_dir))
        ensure_dir(self.wardrobe_storage_dir)
        
        # Initialize wardrobe app (if available)
        self.wardrobe_app = None
        if WardrobeApp:
            try:
                # Create a dummy root for wardrobe app
                dummy_root = tk.Toplevel()
                dummy_root.withdraw()  # Hide it
                self.wardrobe_app = WardrobeApp(dummy_root)
                dummy_root.destroy()  # Clean up
            except Exception as e:
                print(f"Could not initialize wardrobe app: {e}")
        
        self.setup_ui()
        self.refresh_user_list()
    
    def setup_ui(self):
        """Setup the upload tab UI"""
        # Main container
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - User management
        left_panel = ttk.LabelFrame(main_frame, text="User Management", padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # User selection
        ttk.Label(left_panel, text="Select/Create User:").pack(anchor=tk.W)
        user_frame = ttk.Frame(left_panel)
        user_frame.pack(fill=tk.X, pady=5)
        
        self.user_var = tk.StringVar()
        self.user_combo = ttk.Combobox(user_frame, textvariable=self.user_var, state="readonly")
        self.user_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.user_combo.bind("<<ComboboxSelected>>", self.on_user_selected)
        
        ttk.Button(user_frame, text="New User", command=self.create_new_user).pack(side=tk.LEFT, padx=(5, 0))
        
        # User info
        self.user_info_label = ttk.Label(left_panel, text="No user selected", foreground="gray")
        self.user_info_label.pack(pady=5)
        
        # Upload options
        upload_frame = ttk.LabelFrame(left_panel, text="Upload Options", padding="10")
        upload_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(upload_frame, text="📷 Start Camera Capture", 
                  command=self.start_camera_capture).pack(fill=tk.X, pady=2)
        ttk.Button(upload_frame, text="📁 Upload Images", 
                  command=self.upload_images).pack(fill=tk.X, pady=2)
        ttk.Button(upload_frame, text="🔄 Refresh User List", 
                  command=self.refresh_user_list).pack(fill=tk.X, pady=2)
        
        # User wardrobe preview
        preview_frame = ttk.LabelFrame(left_panel, text="User Wardrobe", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Scrollable list
        canvas = tk.Canvas(preview_frame, height=200)
        scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=canvas.yview)
        self.preview_scrollable = ttk.Frame(canvas)
        
        self.preview_scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.preview_scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.wardrobe_list_frame = self.preview_scrollable
        
        # Right panel - Camera/Upload area
        right_panel = ttk.LabelFrame(main_frame, text="Camera/Upload", padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Camera display
        self.camera_label = ttk.Label(right_panel, text="Camera feed will appear here", 
                                     background="black", foreground="white")
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Status
        self.status_label = ttk.Label(right_panel, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(fill=tk.X, pady=(10, 0))
    
    def refresh_user_list(self):
        """Refresh the list of users"""
        users = []
        if os.path.exists(self.wardrobe_storage_dir):
            users = [d for d in os.listdir(self.wardrobe_storage_dir) 
                    if os.path.isdir(os.path.join(self.wardrobe_storage_dir, d))]
        
        self.user_combo['values'] = users
        if users and not self.user_var.get():
            self.user_var.set(users[0])
            self.on_user_selected()
    
    def create_new_user(self):
        """Create a new user with name and gender"""
        # Create a dialog window for name and gender
        dialog = tk.Toplevel(self.parent)
        dialog.title("Create New User")
        dialog.geometry("400x200")
        dialog.transient(self.parent)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Name field
        ttk.Label(dialog, text="Enter user name:").pack(pady=10)
        name_var = tk.StringVar()
        name_entry = ttk.Entry(dialog, textvariable=name_var, width=30)
        name_entry.pack(pady=5)
        name_entry.focus()
        
        # Gender field
        ttk.Label(dialog, text="Select gender:").pack(pady=10)
        gender_var = tk.StringVar(value="M")
        gender_frame = ttk.Frame(dialog)
        gender_frame.pack(pady=5)
        ttk.Radiobutton(gender_frame, text="Male (M)", variable=gender_var, value="M").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(gender_frame, text="Female (F)", variable=gender_var, value="F").pack(side=tk.LEFT, padx=10)
        
        result = [None]  # Use list to store result from nested function
        
        def create_user():
            name = name_var.get().strip()
            gender = gender_var.get()
            
            if not name:
                messagebox.showwarning("Warning", "Please enter a user name")
                return
            
            # Create name with gender suffix
            name_with_gender = f"{name}_{gender}"
            user_dir = os.path.join(self.wardrobe_storage_dir, name_with_gender)
            os.makedirs(user_dir, exist_ok=True)
            
            result[0] = name_with_gender
            dialog.destroy()
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        ttk.Button(button_frame, text="Create", command=create_user).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Bind Enter key to create
        name_entry.bind("<Return>", lambda e: create_user())
        
        # Wait for dialog to close
        dialog.wait_window()
        
        if result[0]:
            name_with_gender = result[0]
            self.refresh_user_list()
            self.user_var.set(name_with_gender)
            self.on_user_selected()
            messagebox.showinfo("Success", f"User '{name_with_gender}' created!")
    
    def on_user_selected(self, event=None):
        """Handle user selection"""
        user = self.user_var.get()
        if user:
            user_dir = os.path.join(self.wardrobe_storage_dir, user)
            if os.path.exists(user_dir):
                # Count images
                images = list(Path(user_dir).glob("*.jpg")) + list(Path(user_dir).glob("*.png"))
                
                # Extract gender from user name
                gender = self._extract_gender_from_name(user)
                gender_display = f"Gender: {gender}" if gender else ""
                
                self.user_info_label.config(
                    text=f"User: {user}\n{gender_display}\nImages: {len(images)}",
                    foreground="black"
                )
                self.update_wardrobe_preview(user_dir)
    
    def _extract_gender_from_name(self, name):
        """Extract gender from user name (format: name_M or name_F)"""
        if name.endswith("_M"):
            return "Male (M)"
        elif name.endswith("_F"):
            return "Female (F)"
        return None
    
    def update_wardrobe_preview(self, user_dir):
        """Update wardrobe preview list"""
        # Clear existing
        for widget in self.wardrobe_list_frame.winfo_children():
            widget.destroy()
        
        # Add images
        images = list(Path(user_dir).glob("*.jpg")) + list(Path(user_dir).glob("*.png"))
        for img_path in images[:20]:  # Show first 20
            img_name = img_path.name
            ttk.Label(self.wardrobe_list_frame, text=img_name).pack(anchor=tk.W)
    
    def start_camera_capture(self):
        """Start camera capture using wardrobe app"""
        if not self.user_var.get():
            messagebox.showwarning("Warning", "Please select or create a user first")
            return
        
        if not self.wardrobe_app:
            messagebox.showerror("Error", "Wardrobe upload system not available")
            return
        
        # Open wardrobe app in new window
        camera_window = tk.Toplevel(self.parent)
        camera_window.title(f"Camera Capture - {self.user_var.get()}")
        camera_window.geometry("1000x700")
        
        # Create wardrobe app instance for this window
        try:
            wardrobe_instance = WardrobeApp(camera_window)
            wardrobe_instance.current_person = self.user_var.get()
            wardrobe_instance.person_folder = os.path.join(
                self.wardrobe_storage_dir, self.user_var.get()
            )
            wardrobe_instance.current_person_label.config(
                text=f"Current: {self.user_var.get()}", 
                foreground="green"
            )
            messagebox.showinfo("Camera", "Camera window opened. Use it to capture clothing items.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not start camera: {e}")
            camera_window.destroy()
    
    def upload_images(self):
        """Upload images from files"""
        if not self.user_var.get():
            messagebox.showwarning("Warning", "Please select or create a user first")
            return
        
        files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG")]
        )
        
        if files:
            user_dir = os.path.join(self.wardrobe_storage_dir, self.user_var.get())
            os.makedirs(user_dir, exist_ok=True)
            
            copied = 0
            for file in files:
                try:
                    filename = os.path.basename(file)
                    dest = os.path.join(user_dir, filename)
                    # Avoid overwriting
                    if os.path.exists(dest):
                        name, ext = os.path.splitext(filename)
                        dest = os.path.join(user_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}")
                    
                    import shutil
                    shutil.copy2(file, dest)
                    copied += 1
                except Exception as e:
                    print(f"Error copying {file}: {e}")
            
            messagebox.showinfo("Success", f"Uploaded {copied} image(s)")
            self.on_user_selected()


class RecommendationsTab:
    """Tab for viewing recommendations"""
    
    def __init__(self, parent, wardrobe_storage_dir=None):
        self.parent = parent
        # Use path_utils to get cross-platform wardrobe storage directory
        if wardrobe_storage_dir is None:
            self.wardrobe_storage_dir = str(get_wardrobe_storage_dir())
        else:
            self.wardrobe_storage_dir = str(normalize_path(wardrobe_storage_dir))
        ensure_dir(self.wardrobe_storage_dir)
        self.recommender = None
        self.current_recommendations = None
        self.initialization_attempted = False
        
        self.setup_ui()
        self.refresh_user_list()
        
        # Try to initialize in background (non-blocking)
        self.parent.after(100, self.initialize_recommender)
    
    def initialize_recommender(self, show_error=True, force_retry=False):
        """Initialize the recommender system"""
        global Recommender  # Declare global at the top
        
        if self.recommender is not None:
            return True  # Already initialized
        
        # Allow retry if force_retry is True, otherwise skip if already attempted
        if self.initialization_attempted and not force_retry:
            return False  # Already tried
        
        self.initialization_attempted = True
        
        try:
            # Try to import Recommender if not already imported
            if Recommender is None:
                try:
                    # Try importing again
                    import importlib.util
                    current_dir = Path(__file__).parent.absolute()
                    recommender_path = current_dir / "recommender.py"
                    
                    if recommender_path.exists():
                        spec = importlib.util.spec_from_file_location("recommender", recommender_path)
                        recommender_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(recommender_module)
                        Recommender = recommender_module.Recommender
                        print("✓ Recommender imported successfully")
                    else:
                        # Try standard import
                        from recommender import Recommender
                        print("✓ Recommender imported successfully")
                except Exception as import_error:
                    if show_error:
                        error_msg = (
                            "Recommender module not found!\n\n"
                            f"Error: {import_error}\n\n"
                            f"Looking for: {recommender_path if 'recommender_path' in locals() else 'recommender.py'}\n\n"
                            "Please ensure:\n"
                            "1. recommender.py exists in recommender_system folder\n"
                            "2. All dependencies are installed\n"
                            "3. Python can import the module"
                        )
                        messagebox.showerror("Error", error_msg)
                    return False
            
            self.status_label.config(text="Initializing recommender...")
            self.parent.update()
            
            self.recommender = Recommender()
            print("✓ Recommender initialized successfully")
            self.status_label.config(text="Recommender ready")
            return True
            
        except FileNotFoundError as e:
            error_msg = str(e)
            if "resnet50_features_pca512.npy" in error_msg or "resnet50_metadata.csv" in error_msg:
                detailed_msg = (
                    "Feature extraction files not found!\n\n"
                    "Required files:\n"
                    "- resnet50_features_pca512.npy\n"
                    "- resnet50_metadata.csv\n\n"
                    "Please run feature extraction first:\n"
                    "cd extracted_features\n"
                    "python feature_extractor.py"
                )
            else:
                detailed_msg = f"File not found: {error_msg}"
            
            if show_error:
                messagebox.showerror("Initialization Error", detailed_msg)
            self.status_label.config(text="Recommender initialization failed - check files")
            # Reset flag to allow retry
            if force_retry:
                self.initialization_attempted = False
            return False
            
        except Exception as e:
            error_msg = f"Could not initialize recommender: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            if show_error:
                messagebox.showerror("Initialization Error", 
                    f"{error_msg}\n\n"
                    "Please check:\n"
                    "1. Feature extraction is complete\n"
                    "2. Required files exist:\n"
                    "   - resnet50_features_pca512.npy\n"
                    "   - resnet50_metadata.csv\n"
                    "3. All dependencies are installed")
            self.status_label.config(text="Recommender initialization failed")
            # Reset flag to allow retry
            if force_retry:
                self.initialization_attempted = False
            return False
    
    def setup_ui(self):
        """Setup the recommendations tab UI"""
        # Main container
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top panel - User selection and controls
        top_panel = ttk.Frame(main_frame)
        top_panel.pack(fill=tk.X, pady=(0, 10))
        
        # User selection
        ttk.Label(top_panel, text="Select User:").pack(side=tk.LEFT, padx=(0, 5))
        self.user_var = tk.StringVar()
        self.user_combo = ttk.Combobox(top_panel, textvariable=self.user_var, 
                                      state="readonly", width=20)
        self.user_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.user_combo.bind("<<ComboboxSelected>>", self.on_user_selected)
        
        # Get recommendations button
        ttk.Button(top_panel, text="Get Recommendations", 
                  command=self.get_recommendations).pack(side=tk.LEFT, padx=(0, 10))
        
        # Strategy selection
        ttk.Label(top_panel, text="Strategy:").pack(side=tk.LEFT, padx=(10, 5))
        self.strategy_var = tk.StringVar(value="hybrid")
        strategy_combo = ttk.Combobox(top_panel, textvariable=self.strategy_var,
                                     values=["hybrid", "similar", "complementary", "category_expansion"],
                                     state="readonly", width=15)
        strategy_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # Number of recommendations
        ttk.Label(top_panel, text="Top K:").pack(side=tk.LEFT, padx=(10, 5))
        self.top_k_var = tk.StringVar(value="20")
        ttk.Spinbox(top_panel, from_=5, to=50, textvariable=self.top_k_var, 
                   width=5).pack(side=tk.LEFT)
        
        # Refresh button
        ttk.Button(top_panel, text="🔄 Refresh", 
                  command=self.refresh_user_list).pack(side=tk.LEFT, padx=(10, 0))
        
        # Initialize button (for manual retry)
        ttk.Button(top_panel, text="🔧 Init Recommender", 
                  command=lambda: self.initialize_recommender(show_error=True, force_retry=True)).pack(side=tk.LEFT, padx=(5, 0))
        
        # Status label
        self.status_label = ttk.Label(top_panel, text="Initializing...", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Recommendations display area
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable canvas for recommendations
        canvas = tk.Canvas(display_frame, bg="white")
        scrollbar = ttk.Scrollbar(display_frame, orient="vertical", command=canvas.yview)
        self.recommendations_frame = ttk.Frame(canvas)
        
        self.recommendations_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.recommendations_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas = canvas
    
    def refresh_user_list(self):
        """Refresh the list of users"""
        users = []
        if os.path.exists(self.wardrobe_storage_dir):
            users = [d for d in os.listdir(self.wardrobe_storage_dir) 
                    if os.path.isdir(os.path.join(self.wardrobe_storage_dir, d))]
        
        self.user_combo['values'] = users
        if users and not self.user_var.get():
            self.user_var.set(users[0])
    
    def on_user_selected(self, event=None):
        """Handle user selection"""
        user = self.user_var.get()
        if user:
            user_dir = os.path.join(self.wardrobe_storage_dir, user)
            if os.path.exists(user_dir):
                images = list(Path(user_dir).glob("*.jpg")) + list(Path(user_dir).glob("*.png"))
                gender = self._extract_gender_from_name(user)
                gender_display = f" | Gender: {gender}" if gender else ""
                self.status_label.config(text=f"User: {user}{gender_display} | {len(images)} images in wardrobe")
    
    def _extract_gender_from_name(self, name):
        """Extract gender from user name (format: name_M or name_F)"""
        if name.endswith("_M"):
            return "Male (M)"
        elif name.endswith("_F"):
            return "Female (F)"
        return None
    
    def _get_gender_filter(self, name):
        """Get gender filter value for recommender (M->'Men', F->'Women')"""
        if name.endswith("_M"):
            return "Men"
        elif name.endswith("_F"):
            return "Women"
        return None
    
    def get_recommendations(self):
        """Get recommendations for selected user"""
        user = self.user_var.get()
        if not user:
            messagebox.showwarning("Warning", "Please select a user")
            return
        
        # Auto-initialize recommender if not already initialized
        if not self.recommender:
            self.status_label.config(text="Initializing recommender...")
            self.parent.update()
            
            # Try to initialize (allow retry if previous attempt failed)
            if not self.initialize_recommender(show_error=True, force_retry=True):
                # Initialization failed, don't proceed
                self.status_label.config(text="Initialization failed - click 'Init Recommender' to retry")
                return
        
        user_dir = os.path.join(self.wardrobe_storage_dir, user)
        if not os.path.exists(user_dir):
            messagebox.showerror("Error", f"User directory not found: {user_dir}")
            return
        
        # Get user's wardrobe images
        image_paths = list(Path(user_dir).glob("*.jpg")) + list(Path(user_dir).glob("*.png"))
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            messagebox.showwarning("Warning", f"No images found for user '{user}'")
            return
        
        # Get recommendations
        try:
            self.status_label.config(text="Getting recommendations...")
            self.parent.update()
            
            strategy = self.strategy_var.get()
            top_k = int(self.top_k_var.get())
            
            # Extract gender from user name and create filter
            gender_filter_value = self._get_gender_filter(user)
            filters = None
            if gender_filter_value:
                filters = {'gender': [gender_filter_value]}
                self.status_label.config(text=f"Getting recommendations (filtered for {gender_filter_value})...")
                self.parent.update()
            
            recommendations = self.recommender.get_recommendations(
                user_wardrobe_paths=image_paths,
                strategy=strategy,
                top_k=top_k,
                filters=filters
            )
            
            self.current_recommendations = recommendations
            self.display_recommendations(recommendations)
            
            self.status_label.config(text=f"Found {len(recommendations)} recommendations")
            
        except FileNotFoundError as e:
            error_msg = str(e)
            if "resnet50" in error_msg.lower() or "metadata" in error_msg.lower():
                detailed_msg = (
                    "Required files not found!\n\n"
                    "Please ensure feature extraction is complete:\n"
                    "- resnet50_features_pca512.npy\n"
                    "- resnet50_metadata.csv\n\n"
                    "Run: cd extracted_features && python feature_extractor.py"
                )
            else:
                detailed_msg = f"File not found: {error_msg}"
            
            messagebox.showerror("Error", detailed_msg)
            self.status_label.config(text="Error: Required files not found")
            
        except Exception as e:
            error_msg = f"Error getting recommendations: {e}"
            messagebox.showerror("Error", error_msg)
            import traceback
            traceback.print_exc()
            self.status_label.config(text="Error getting recommendations")
    
    def display_recommendations(self, recommendations):
        """Display recommendations with images and details"""
        # Clear existing
        for widget in self.recommendations_frame.winfo_children():
            widget.destroy()
        
        if recommendations is None or len(recommendations) == 0:
            ttk.Label(self.recommendations_frame, 
                     text="No recommendations found",
                     font=("Arial", 12)).pack(pady=20)
            return
        
        # Display each recommendation
        for idx, (_, item) in enumerate(recommendations.iterrows(), 1):
            self.create_recommendation_card(item, idx)
    
    def create_recommendation_card(self, item, index):
        """Create a card for a single recommendation"""
        card_frame = ttk.Frame(self.recommendations_frame, relief=tk.RAISED, borderwidth=2)
        card_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Left side - Image
        image_frame = ttk.Frame(card_frame)
        image_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Try to load product image
        image_path = self.get_product_image_path(item.get('id', ''))
        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                img.thumbnail((150, 150), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                img_label = ttk.Label(image_frame, image=photo)
                img_label.image = photo  # Keep a reference
                img_label.pack()
            except Exception as e:
                ttk.Label(image_frame, text="No Image", width=20, 
                         background="lightgray").pack()
        else:
            ttk.Label(image_frame, text="No Image", width=20, 
                     background="lightgray").pack()
        
        # Right side - Details
        details_frame = ttk.Frame(card_frame)
        details_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title = item.get('productDisplayName', f"Product {item.get('id', 'N/A')}")
        title_label = ttk.Label(details_frame, text=title, 
                               font=("Arial", 12, "bold"))
        title_label.pack(anchor=tk.W)
        
        # Details
        details_text = f"""
Rank: #{index}
ID: {item.get('id', 'N/A')}
Type: {item.get('articleType', 'N/A')}
Category: {item.get('masterCategory', 'N/A')}
Color: {item.get('baseColour', 'N/A')}
Gender: {item.get('gender', 'N/A')}
Similarity: {item.get('similarity_score', 0):.3f}
        """.strip()
        
        details_label = ttk.Label(details_frame, text=details_text, 
                                 justify=tk.LEFT, font=("Arial", 9))
        details_label.pack(anchor=tk.W, pady=5)
        
        # Recommendation reason
        reason = item.get('recommendation_reason', 'N/A')
        reason_label = ttk.Label(details_frame, text=f"Reason: {reason}", 
                                font=("Arial", 8), foreground="gray")
        reason_label.pack(anchor=tk.W)
    
    def get_product_image_path(self, product_id):
        """Get the path to a product image"""
        # Use path_utils to find product images directory
        images_dir = get_product_images_dir()
        
        if images_dir is None:
            return None
        
        # Try different extensions
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        for ext in extensions:
            img_path = images_dir / f"{product_id}{ext}"
            if img_path.exists():
                return str(img_path)
        
        # If not found, return None (will show placeholder)
        return None


class WardrobeRecommenderApp:
    """Main application combining wardrobe upload and recommendations"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Wardrobe Recommender System")
        self.root.geometry("1400x900")
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Wardrobe Upload
        self.upload_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.upload_tab, text="📁 Wardrobe Upload")
        self.upload_panel = WardrobeUploadTab(
            self.upload_tab,
            wardrobe_storage_dir=None  # Will use path_utils to find/create
        )
        
        # Tab 2: Recommendations
        self.recommendations_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.recommendations_tab, text="🎯 Recommendations")
        self.recommendations_panel = RecommendationsTab(
            self.recommendations_tab,
            wardrobe_storage_dir=None  # Will use path_utils to find/create
        )
        
        # Menu bar
        self.create_menu()
    
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About", 
            "Wardrobe Recommender System\n\n"
            "Upload wardrobe items for multiple users and get personalized recommendations!")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = WardrobeRecommenderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

