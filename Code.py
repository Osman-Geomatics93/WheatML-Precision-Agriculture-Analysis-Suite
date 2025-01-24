import tkinter as tk
from ttkbootstrap import Style, ttk
import ttkbootstrap as tb
from tkinter import messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
from PIL import Image, ImageTk

# Define paths
APP_DIR = r"C:\Users\Osman-Ibrahim\OneDrive\سطح المكتب\Yield_Application"
IMAGE_PATH = os.path.join(APP_DIR, "Application_Image.png")
TRAIN_DATA_PATH = os.path.join(APP_DIR, "Training_Data.csv")
TEST_DATA_PATH = os.path.join(APP_DIR, "Test_Data.csv")

class LoginWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("Login - Gezira Irrigation Scheme")
        self.master.geometry("400x500")
        
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x = (screen_width - 400) // 2
        y = (screen_height - 500) // 2
        self.master.geometry(f"400x500+{x}+{y}")

        style = Style(theme='cosmo')
        
        self.frame = ttk.Frame(master, padding="20")
        self.frame.pack(fill=tk.BOTH, expand=True)

        try:
            if os.path.exists(IMAGE_PATH):
                img = Image.open(IMAGE_PATH)
                img = img.resize((200, 200), Image.Resampling.LANCZOS)
                self.logo = ImageTk.PhotoImage(img)
                logo_label = ttk.Label(self.frame, image=self.logo)
                logo_label.pack(pady=20)
        except Exception as e:
            print(f"Error loading image: {e}")

        title_label = ttk.Label(self.frame, text="Gezira Irrigation Scheme", 
                               font=("Helvetica", 16, "bold"))
        title_label.pack(pady=10)

        username_frame = ttk.Frame(self.frame)
        username_frame.pack(fill=tk.X, pady=5)
        ttk.Label(username_frame, text="Username:").pack(anchor=tk.W)
        self.username_entry = ttk.Entry(username_frame)
        self.username_entry.pack(fill=tk.X)

        password_frame = ttk.Frame(self.frame)
        password_frame.pack(fill=tk.X, pady=5)
        ttk.Label(password_frame, text="Password:").pack(anchor=tk.W)
        self.password_entry = ttk.Entry(password_frame, show="*")
        self.password_entry.pack(fill=tk.X)

        self.login_button = ttk.Button(self.frame, text="Login", 
                                     command=self.login, style='success.TButton')
        self.login_button.pack(pady=20)

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        if username == "HRC_Sudan" and password == "422436":
            self.master.withdraw()
            app_window = tk.Toplevel()
            app = WheatYieldPredictionApp(app_window)
            app_window.protocol("WM_DELETE_WINDOW", lambda: self.on_closing(app_window))
        else:
            messagebox.showerror("Error", "Invalid username or password")

    def on_closing(self, app_window):
        app_window.destroy()
        self.master.destroy()

class WheatYieldPredictionApp:
    def __init__(self, master):
        self.master = master
        self.style = Style(theme='cosmo')
        self.master.title("Wheat Yield and Water Productivity Prediction")
        self.master.geometry("1200x800")
        
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x = (screen_width - 1200) // 2
        y = (screen_height - 800) // 2
        self.master.geometry(f"1200x800+{x}+{y}")

        self.scaler = StandardScaler()
        self.load_and_prepare_data()
        self.create_widgets()
        self.train_models()

    def load_and_prepare_data(self):
        self.train_df = pd.read_csv(TRAIN_DATA_PATH)
        self.test_df = pd.read_csv(TEST_DATA_PATH)

        columns_to_remove = ['classvalue', 'sample']
        self.train_df = self.train_df.drop(columns=columns_to_remove, errors='ignore')
        self.test_df = self.test_df.drop(columns=columns_to_remove, errors='ignore')

        if 'Calculated Yield ton/feddan' in self.train_df.columns:
            self.train_df = self.train_df.rename(columns={'Calculated Yield ton/feddan': 'Calculated Yield ton/ha'})
        if 'Calculated Yield ton/feddan' in self.test_df.columns:
            self.test_df = self.test_df.rename(columns={'Calculated Yield ton/feddan': 'Calculated Yield ton/ha'})

        self.yield_column = 'Real Yield ton/feddan'
        self.wpy_column = 'Wpy'
        self.features = [col for col in self.train_df.columns if col not in [self.yield_column, self.wpy_column]]

        self.X_train = self.train_df[self.features]
        self.y_yield_train = self.train_df[self.yield_column]
        self.y_wpy_train = self.train_df[self.wpy_column]

        self.X_test = self.test_df[self.features]
        self.y_yield_test = self.test_df[self.yield_column]
        self.y_wpy_test = self.test_df[self.wpy_column]

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.input_tab = ttk.Frame(self.notebook)
        self.viz_tab = ttk.Frame(self.notebook)
        self.heatmap_tab = ttk.Frame(self.notebook)
        self.feature_importance_tab = ttk.Frame(self.notebook)
        self.about_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.input_tab, text="Input & Predict")
        self.notebook.add(self.viz_tab, text="Performance Visualization")
        self.notebook.add(self.heatmap_tab, text="Correlation Heatmap")
        self.notebook.add(self.feature_importance_tab, text="Feature Importance")
        self.notebook.add(self.about_tab, text="About")

        self.create_input_widgets()
        self.create_viz_widgets()
        self.create_heatmap_widgets()
        self.create_feature_importance_widgets()
        self.create_about_widgets()

    def create_input_widgets(self):
        input_frame = ttk.Frame(self.input_tab, padding=20)
        input_frame.pack(fill=tk.BOTH, expand=True)

        try:
            if os.path.exists(IMAGE_PATH):
                img = Image.open(IMAGE_PATH)
                img = img.resize((150, 150), Image.Resampling.LANCZOS)
                self.logo = ImageTk.PhotoImage(img)
                logo_label = ttk.Label(input_frame, image=self.logo)
                logo_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        except Exception as e:
            print(f"Error loading image: {e}")

        title_label = ttk.Label(input_frame, text="Wheat Yield Prediction", 
                               font=("Helvetica", 16, "bold"))
        title_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))

        canvas = tk.Canvas(input_frame)
        scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        self.feature_entries = {}
        for i, feature in enumerate(self.features):
            ttk.Label(scrollable_frame, text=f"{feature}:").grid(row=i, column=0, padx=5, pady=5, sticky='e')
            self.feature_entries[feature] = ttk.Entry(scrollable_frame)
            self.feature_entries[feature].grid(row=i, column=1, padx=5, pady=5, sticky='ew')

        canvas.grid(row=2, column=0, sticky="nsew")
        scrollbar.grid(row=2, column=1, sticky="ns")
        input_frame.grid_columnconfigure(0, weight=1)
        input_frame.grid_rowconfigure(2, weight=1)

        control_frame = ttk.Frame(input_frame)
        control_frame.grid(row=3, column=0, columnspan=2, pady=10)

        ttk.Label(control_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="Linear Regression")
        model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, 
                                   values=["Linear Regression", "Random Forest", "Gradient Boosting", 
                                           "XGBoost", "KNN", "Decision Tree", "Bagging Regressor"])
        model_combo.pack(side=tk.LEFT, padx=5)

        self.predict_button = ttk.Button(control_frame, text="Predict", 
                                       command=self.animate_prediction, style='success.TButton')
        self.predict_button.pack(side=tk.LEFT, padx=5)

        self.progress_bar = ttk.Progressbar(input_frame, mode='indeterminate')
        self.progress_bar.grid(row=4, column=0, columnspan=2, pady=10, sticky='ew')

        self.result_label = ttk.Label(input_frame, text="", font=("Helvetica", 12))
        self.result_label.grid(row=5, column=0, columnspan=2, pady=10)

        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=10)

        self.explain_button = ttk.Button(button_frame, text="Explain Parameters", 
                                       command=self.explain_parameters, style='info.TButton')
        self.explain_button.pack(side=tk.LEFT, padx=5)

        self.save_model_button = ttk.Button(button_frame, text="Save Model", 
                                          command=self.save_model, style='info.TButton')
        self.save_model_button.pack(side=tk.LEFT, padx=5)

        self.load_model_button = ttk.Button(button_frame, text="Load Model", 
                                          command=self.load_model, style='info.TButton')
        self.load_model_button.pack(side=tk.LEFT, padx=5)

    def create_viz_widgets(self):
        self.viz_canvas = FigureCanvasTkAgg(plt.Figure(figsize=(10, 6)), master=self.viz_tab)
        self.viz_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_heatmap_widgets(self):
        self.heatmap_canvas = FigureCanvasTkAgg(plt.Figure(figsize=(10, 8)), master=self.heatmap_tab)
        self.heatmap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_feature_importance_widgets(self):
        self.feature_importance_canvas = FigureCanvasTkAgg(plt.Figure(figsize=(10, 6)), 
                                                         master=self.feature_importance_tab)
        self.feature_importance_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_about_widgets(self):
        about_frame = ttk.Frame(self.about_tab, padding=20)
        about_frame.pack(fill=tk.BOTH, expand=True)

        try:
            if os.path.exists(IMAGE_PATH):
                img = Image.open(IMAGE_PATH)
                img = img.resize((200, 200), Image.Resampling.LANCZOS)
                self.about_logo = ImageTk.PhotoImage(img)
                about_logo_label = ttk.Label(about_frame, image=self.about_logo)
                about_logo_label.pack(pady=(0, 20))
        except Exception as e:
            print(f"Error loading image: {e}")

        title_label = ttk.Label(about_frame, text="MSc Project", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=(0, 10))

        project_title = """Application of Remote Sensing and Machine Learning for Estimating Crops Areas, 
        Yield, and Water Productivity of Wheat in the Gezira Irrigation Scheme"""
        project_label = ttk.Label(about_frame, text=project_title, wraplength=600, justify=tk.CENTER)
        project_label.pack(pady=(0, 20))

        info_text = """
        Supervisor: Assoc. Prof. VOLKAN YILMAZ
        Student: Osman Osama Ahmed Ibrahim
        University: Karadeniz Technical University - Turkey-2024
        Auxiliary Research Center: Hydraulics Research Center - Sudan-2024
        """
        info_label = ttk.Label(about_frame, text=info_text, justify=tk.LEFT)
        info_label.pack(pady=(0, 20))

    def train_models(self):
        self.models = {
            "Linear Regression": (LinearRegression(), LinearRegression()),
            "Random Forest": (RandomForestRegressor(n_estimators=100, random_state=42), 
                              RandomForestRegressor(n_estimators=100, random_state=42)),
            "Gradient Boosting": (GradientBoostingRegressor(n_estimators=100, random_state=42), 
                                  GradientBoostingRegressor(n_estimators=100, random_state=42)),
            "XGBoost": (XGBRegressor(n_estimators=100, random_state=42), 
                        XGBRegressor(n_estimators=100, random_state=42)),
            "KNN": (KNeighborsRegressor(), KNeighborsRegressor()),
            "Decision Tree": (DecisionTreeRegressor(random_state=42), DecisionTreeRegressor(random_state=42)),
            "Bagging Regressor": (BaggingRegressor(n_estimators=100, random_state=42), 
                                  BaggingRegressor(n_estimators=100, random_state=42))
        }

        self.model_metrics = {}
        for name, (yield_model, wpy_model) in self.models.items():
            yield_model.fit(self.X_train_scaled, self.y_yield_train)
            wpy_model.fit(self.X_train_scaled, self.y_wpy_train)
            
            y_yield_pred = yield_model.predict(self.X_test_scaled)
            y_wpy_pred = wpy_model.predict(self.X_test_scaled)

            self.model_metrics[name] = {
                "yield_r2": r2_score(self.y_yield_test, y_yield_pred),
                "yield_mae": mean_absolute_error(self.y_yield_test, y_yield_pred),
                "yield_rmse": np.sqrt(mean_squared_error(self.y_yield_test, y_yield_pred)),
                "wpy_r2": r2_score(self.y_wpy_test, y_wpy_pred),
                "wpy_mae": mean_absolute_error(self.y_wpy_test, y_wpy_pred),
                "wpy_rmse": np.sqrt(mean_squared_error(self.y_wpy_test, y_wpy_pred))
            }

        self.visualize_model_performance()
        self.create_correlation_heatmap()
        self.visualize_feature_importance()

    def animate_prediction(self):
        self.progress_bar.start(10)
        self.predict_button.config(state='disabled')
        self.master.after(1000, self.predict)

    def predict(self):
        try:
            input_data = [float(self.feature_entries[feature].get()) for feature in self.features]
            input_scaled = self.scaler.transform([input_data])

            model_name = self.model_var.get()
            yield_model, wpy_model = self.models[model_name]

            yield_pred = yield_model.predict(input_scaled)[0]
            wpy_pred = wpy_model.predict(input_scaled)[0]

            result_text = f"Predicted Yield: {yield_pred:.2f} ton/ha\n"
            result_text += f"Predicted Water Productivity: {wpy_pred:.4f}\n\n"
            result_text += f"Model: {model_name}\n"
            result_text += f"Yield R²: {self.model_metrics[model_name]['yield_r2']:.4f}\n"
            result_text += f"WPy R²: {self.model_metrics[model_name]['wpy_r2']:.4f}"

            self.result_label.config(text=result_text)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred: {str(e)}")
        finally:
            self.progress_bar.stop()
            self.predict_button.config(state='normal')

    def visualize_model_performance(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        models = list(self.model_metrics.keys())
        yield_r2 = [metrics['yield_r2'] for metrics in self.model_metrics.values()]
        wpy_r2 = [metrics['wpy_r2'] for metrics in self.model_metrics.values()]

        x = range(len(models))
        width = 0.35

        ax1.bar(x, yield_r2, width, label='Yield', color='skyblue')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Yield Prediction Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()

        ax2.bar(x, wpy_r2, width, label='Water Productivity', color='lightgreen')
        ax2.set_ylabel('R² Score')
        ax2.set_title('Water Productivity Prediction Performance')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend()

        plt.tight_layout()
        self.viz_canvas.figure = fig
        self.viz_canvas.draw()

    def create_correlation_heatmap(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = self.train_df[self.features + [self.wpy_column]].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap')
        plt.tight_layout()
        self.heatmap_canvas.figure = fig
        self.heatmap_canvas.draw()

    def visualize_feature_importance(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        rf_yield_model = self.models["Random Forest"][0]
        rf_wpy_model = self.models["Random Forest"][1]
        
        yield_importance = rf_yield_model.feature_importances_
        wpy_importance = rf_wpy_model.feature_importances_
        
        yield_indices = np.argsort(yield_importance)[::-1]
        wpy_indices = np.argsort(wpy_importance)[::-1]
        
        ax1.bar(range(len(yield_importance)), yield_importance[yield_indices], align="center")
        ax1.set_xticks(range(len(yield_importance)))
        ax1.set_xticklabels([self.features[i] for i in yield_indices], rotation=90)
        ax1.set_title("Feature Importance for Yield Prediction")
        ax1.set_ylabel("Importance")
        
        ax2.bar(range(len(wpy_importance)), wpy_importance[wpy_indices], align="center")
        ax2.set_xticks(range(len(wpy_importance)))
        ax2.set_xticklabels([self.features[i] for i in wpy_indices], rotation=90)
        ax2.set_title("Feature Importance for Water Productivity Prediction")
        ax2.set_ylabel("Importance")
        
        plt.tight_layout()
        self.feature_importance_canvas.figure = fig
        self.feature_importance_canvas.draw()

    def explain_parameters(self):
        explanation = """
        Parameter Explanations:

        AETI: Actual Evapotranspiration (mm)
        NPP: Net Primary Production (kg/m²)
        T: Transpiration (mm)
        Adequacy: Ratio of actual to potential evapotranspiration
        BF: Beneficial Fraction 
        AGBM: Above Ground Biomass (ton/ha)
        WPb: Biomass Water Productivity (kg/m³)
        NDVI: Normalized Difference Vegetation Index
        EVI: Enhanced Vegetation Index
        SIPI: Structure Insensitive Pigment Index
        Wpy: Water Productivity (kg/m³)
        """
        messagebox.showinfo("Parameter Explanations", explanation)

    def save_model(self):
        model_name = self.model_var.get()
        yield_model, wpy_model = self.models[model_name]
        
        file_path = filedialog.asksaveasfilename(defaultextension=".joblib",
                                                 filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")])
        if file_path:
            model_data = {
                'yield_model': yield_model,
                'wpy_model': wpy_model,
                'scaler': self.scaler,
                'features': self.features
            }
            joblib.dump(model_data, file_path)
            messagebox.showinfo("Save Model", f"Model saved successfully to {file_path}")

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")])
        if file_path:
            try:
                model_data = joblib.load(file_path)
                self.models[self.model_var.get()] = (model_data['yield_model'], model_data['wpy_model'])
                self.scaler = model_data['scaler']
                self.features = model_data['features']
                messagebox.showinfo("Load Model", f"Model loaded successfully from {file_path}")
            except Exception as e:
                messagebox.showerror("Load Model Error", f"An error occurred while loading the model: {str(e)}")

if __name__ == "__main__":
    root = tb.Window(themename="cosmo")
    login = LoginWindow(root)
    root.mainloop()