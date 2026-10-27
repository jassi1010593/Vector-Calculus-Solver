import tkinter as tk
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import sympy as sp

# Set UI Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class AdvancedMathUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Advanced Vector Calculus & Analysis Solver")
        self.geometry("1200x800")

        # --- Layout Split ---
        self.grid_columnconfigure(0, weight=1)  # Sidebar
        self.grid_columnconfigure(1, weight=4)  # Main Content
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="Theorem Solver", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(pady=20)

        self.btn_stokes = ctk.CTkButton(self.sidebar, text="Stokes' Theorem", command=lambda: self.switch_tab("Stokes"))
        self.btn_stokes.pack(pady=10, padx=20, fill="x")

        self.btn_greens = ctk.CTkButton(self.sidebar, text="Green's Theorem", command=lambda: self.switch_tab("Greens"))
        self.btn_greens.pack(pady=10, padx=20, fill="x")

        self.btn_gauss = ctk.CTkButton(self.sidebar, text="Gauss Divergence", command=lambda: self.switch_tab("Gauss"))
        self.btn_gauss.pack(pady=10, padx=20, fill="x")

        self.btn_maxmin = ctk.CTkButton(self.sidebar, text="Maxima & Minima", command=lambda: self.switch_tab("MaxMin"))
        self.btn_maxmin.pack(pady=10, padx=20, fill="x")

        # Main Workspace
        self.workspace = ctk.CTkFrame(self)
        self.workspace.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.workspace.grid_columnconfigure(0, weight=1)
        self.workspace.grid_rowconfigure(2, weight=1) # Plot area gets weight

        # Top Control Header
        self.header_label = ctk.CTkLabel(self.workspace, text="Select a Theorem from the left", font=ctk.CTkFont(size=18, weight="bold"))
        self.header_label.grid(row=0, column=0, pady=10, padx=10, sticky="w")

        # Dynamic Input Frame (FIXED: Removed fill="x")
        self.input_frame = ctk.CTkFrame(self.workspace)
        self.input_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        # Output Text/Meaning Frame (FIXED: Removed fill="x")
        self.output_text = ctk.CTkTextbox(self.workspace, height=120, font=ctk.CTkFont(size=13))
        self.output_text.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        # Matplotlib Figure Placeholder (FIXED: Removed fill="both")
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.workspace)
        self.canvas.get_tk_widget().grid(row=2, column=0, shortcut=None, sticky="nsew", padx=10, pady=10)
        
        # Toolbar Frame (FIXED: Removed fill="x")
        self.toolbar_frame = ctk.CTkFrame(self.workspace, height=40)
        self.toolbar_frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        self.current_mode = None

    def clear_input_frame(self):
        for widget in self.input_frame.winfo_children():
            widget.destroy()

    def switch_tab(self, mode):
        self.current_mode = mode
        self.clear_input_frame()
        self.fig.clear()
        self.canvas.draw()
        self.output_text.delete("1.0", tk.END)

        if mode == "Stokes":
            self.header_label.configure(text="Stokes' Theorem Solver (Surface Curl vs Line Integral)")
            self.setup_vector_inputs(include_z=True)
        elif mode == "Greens":
            self.header_label.configure(text="Green's Theorem Solver (2D Flow & Circulation)")
            self.setup_vector_inputs(include_z=False)
        elif mode == "Gauss":
            self.header_label.configure(text="Gauss Divergence Theorem Solver (3D Flux)")
            self.setup_vector_inputs(include_z=True)
        elif mode == "MaxMin":
            self.header_label.configure(text="Maxima and Minima of 2D / 3D Surfaces")
            self.setup_maxmin_inputs()

    def setup_vector_inputs(self, include_z=True):
        ctk.CTkLabel(self.input_frame, text="F_x (i):").grid(row=0, column=0, padx=5, pady=5)
        self.ent_fx = ctk.CTkEntry(self.input_frame, width=150)
        self.ent_fx.insert(0, "-y")
        self.ent_fx.grid(row=0, column=1, padx=5, pady=5)

        ctk.CTkLabel(self.input_frame, text="F_y (j):").grid(row=0, column=2, padx=5, pady=5)
        self.ent_fy = ctk.CTkEntry(self.input_frame, width=150)
        self.ent_fy.insert(0, "x")
        self.ent_fy.grid(row=0, column=3, padx=5, pady=5)

        if include_z:
            ctk.CTkLabel(self.input_frame, text="F_z (k):").grid(row=0, column=4, padx=5, pady=5)
            self.ent_fz = ctk.CTkEntry(self.input_frame, width=150)
            self.ent_fz.insert(0, "z" if self.current_mode == "Gauss" else "0")
            self.ent_fz.grid(row=0, column=5, padx=5, pady=5)

        btn_calc = ctk.CTkButton(self.input_frame, text="Solve & Plot", command=self.process_vector_calculus)
        btn_calc.grid(row=0, column=6, padx=15, pady=5)

    def setup_maxmin_inputs(self):
        ctk.CTkLabel(self.input_frame, text="f(x, y):").grid(row=0, column=0, padx=5, pady=5)
        self.ent_fxy = ctk.CTkEntry(self.input_frame, width=300)
        self.ent_fxy.insert(0, "x**3 + y**3 - 3*x - 12*y")
        self.ent_fxy.grid(row=0, column=1, padx=5, pady=5)

        btn_calc = ctk.CTkButton(self.input_frame, text="Find Extremum & Plot", command=self.process_maxmin)
        btn_calc.grid(row=0, column=2, padx=15, pady=5)

    def process_vector_calculus(self):
        x, y, z = sp.symbols('x y z')
        fx_str = self.ent_fx.get()
        fy_str = self.ent_fy.get()
        
        try:
            F_x = sp.sympify(fx_str)
            F_y = sp.sympify(fy_str)
            F_z = sp.sympify(self.ent_fz.get()) if self.current_mode in ["Stokes", "Gauss"] else 0
        except Exception as e:
            self.output_text.insert(tk.END, f"Input Parsing Error: {e}")
            return

        self.fig.clear()
        self.output_text.delete("1.0", tk.END)

        if self.current_mode == "Greens":
            dq_dx = sp.diff(F_y, x)
            dp_dy = sp.diff(F_x, y)
            integrand = dq_dx - dp_dy
            
            res_str = f"--- Green's Theorem Analytics ---\n"
            res_str += f"Vector Field F = ({F_x})i + ({F_y})j\n"
            res_str += f"Integrand (dQ/dx - dP/dy): {integrand}\n\n"
            res_str += f"PHYSICAL MEANING:\n"
            res_str += f"This calculates the macroscopic circulation of the fluid field around a closed loop.\n"
            res_str += f"If Curl component ({integrand}) > 0, there is a net counter-clockwise fluid rotation inside the area.\n"
            
            self.output_text.insert(tk.END, res_str)
            
            ax = self.fig.add_subplot(111)
            X, Y = np.meshgrid(np.linspace(-3, 3, 15), np.linspace(-3, 3, 15))
            f_x_num = sp.lambdify((x, y), F_x, 'numpy')(X, Y)
            f_y_num = sp.lambdify((x, y), F_y, 'numpy')(X, Y)
            
            if isinstance(f_x_num, (int, float)): f_x_num = np.full_like(X, f_x_num)
            if isinstance(f_y_num, (int, float)): f_y_num = np.full_like(Y, f_y_num)
                
            ax.quiver(X, Y, f_x_num, f_y_num, color='cyan')
            ax.set_title("2D Vector Field Flow & Rotation Visualization")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

        elif self.current_mode == "Stokes":
            curl_x = sp.diff(F_z, y) - sp.diff(F_y, z)
            curl_y = sp.diff(F_x, z) - sp.diff(F_z, x)
            curl_z = sp.diff(F_y, x) - sp.diff(F_x, y)
            
            res_str = f"--- Stokes' Theorem Analytics ---\n"
            res_str += f"Vector Field F = ({F_x})i + ({F_y})j + ({F_z})k\n"
            res_str += f"Curl(F) Vector = ({curl_x})i + ({curl_y})j + ({curl_z})k\n\n"
            res_str += f"PHYSICAL MEANING:\n"
            res_str += f"Stokes' theorem relates the line integral of a vector field around a closed loop to the flux of its curl over an open surface.\n"
            res_str += f"It measures how much the vector field spins or swirls around the boundary of the surface patch.\n"
            
            self.output_text.insert(tk.END, res_str)

            ax = self.fig.add_subplot(111, projection='3d')
            X, Y, Z = np.meshgrid(np.linspace(-2, 2, 8), np.linspace(-2, 2, 8), np.linspace(-2, 2, 8))
            
            fx_n = sp.lambdify((x,y,z), F_x, 'numpy')(X,Y,Z)
            fy_n = sp.lambdify((x,y,z), F_y, 'numpy')(X,Y,Z)
            fz_n = sp.lambdify((x,y,z), F_z, 'numpy')(X,Y,Z)
            
            if isinstance(fx_n, (int,float)): fx_n = np.full_like(X, fx_n)
            if isinstance(fy_n, (int,float)): fy_n = np.full_like(Y, fy_n)
            if isinstance(fz_n, (int,float)): fz_n = np.full_like(Z, fz_n)

            ax.quiver(X, Y, Z, fx_n, fy_n, fz_n, length=0.2, normalize=True, color='deepskyblue', alpha=0.6)
            ax.set_title("3D Vector Field Topology")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        elif self.current_mode == "Gauss":
            div = sp.diff(F_x, x) + sp.diff(F_y, y) + sp.diff(F_z, z)
            
            res_str = f"--- Gauss Divergence Theorem Analytics ---\n"
            res_str += f"Vector Field F = ({F_x})i + ({F_y})j + ({F_z})k\n"
            res_str += f"Divergence (div F) = {div}\n\n"
            res_str += f"PHYSICAL MEANING:\n"
            res_str += f"Divergence measures the net volumetric 'expansion' or 'contraction' of the field from a point.\n"
            res_str += f"If div F > 0, the point acts as a Source (fluid expands outward).\n"
            res_str += f"If div F < 0, the point acts as a Sink (fluid is compressing inward).\n"
            
            self.output_text.insert(tk.END, res_str)

            ax = self.fig.add_subplot(111, projection='3d')
            X, Y, Z = np.meshgrid(np.linspace(-2, 2, 7), np.linspace(-2, 2, 7), np.linspace(-2, 2, 7))
            fx_n = sp.lambdify((x,y,z), F_x, 'numpy')(X,Y,Z)
            fy_n = sp.lambdify((x,y,z), F_y, 'numpy')(X,Y,Z)
            fz_n = sp.lambdify((x,y,z), F_z, 'numpy')(X,Y,Z)
            
            if isinstance(fx_n, (int,float)): fx_n = np.full_like(X, fx_n)
            if isinstance(fy_n, (int,float)): fy_n = np.full_like(Y, fy_n)
            if isinstance(fz_n, (int,float)): fz_n = np.full_like(Z, fz_n)

            ax.quiver(X, Y, Z, fx_n, fy_n, fz_n, length=0.3, color='tomato')
            ax.set_title("3D Fluid Flux Divergence Grid")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        self.canvas.draw()

    def process_maxmin(self):
        x, y = sp.symbols('x y')
        f_str = self.ent_fxy.get()
        
        try:
            f = sp.sympify(f_str)
        except Exception as e:
            self.output_text.insert(tk.END, f"Input Parsing Error: {e}")
            return

        self.fig.clear()
        self.output_text.delete("1.0", tk.END)

        fx = sp.diff(f, x)
        fy = sp.diff(f, y)

        critical_points = sp.solve([fx, fy], (x, y), dict=True)
        
        fxx = sp.diff(fx, x)
        fyy = sp.diff(fy, y)
        fxy = sp.diff(fx, y)

        res_str = f"--- Maxima, Minima & Saddle Points ---\n"
        res_str += f"Function: f(x,y) = {f}\n"
        res_str += f"Found {len(critical_points)} critical point candidates:\n\n"

        valid_pts = []
        for index, pt in enumerate(critical_points):
            if pt[x].is_real and pt[y].is_real:
                px_val = float(pt[x])
                py_val = float(pt[y])
                
                A = float(fxx.subs({x: px_val, y: py_val}))
                B = float(fxy.subs({x: px_val, y: py_val}))
                C = float(fyy.subs({x: px_val, y: py_val}))
                D = A * C - B**2
                
                pz_val = float(f.subs({x: px_val, y: py_val}))
                valid_pts.append((px_val, py_val, pz_val))

                res_str += f"Pt {index+1}: ({px_val:.2f}, {py_val:.2f}) -> f={pz_val:.2f} | "
                if D > 0 and A > 0:
                    res_str += "Classification: Local Minimum\n"
                elif D > 0 and A < 0:
                    res_str += "Classification: Local Maximum\n"
                elif D < 0:
                    res_str += "Classification: Saddle Point\n"
                else:
                    res_str += "Classification: Inconclusive\n"

        res_str += f"\nGEOMETRICAL MEANING:\n"
        res_str += f"Maxima are peaks, Minima are valleys, and Saddle points act like structural mountain passes where slopes change signs along different axes."
        
        self.output_text.insert(tk.END, res_str)

        ax = self.fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(np.linspace(-5, 5, 40), np.linspace(-5, 5, 40))
        
        f_num = sp.lambdify((x, y), f, 'numpy')
        try:
            Z = f_num(X, Y)
            if isinstance(Z, (int, float)): Z = np.full_like(X, Z)
            
            ax.plot_surface(X, Y, Z, cmap='terrain', edgecolor='none', alpha=0.7)
            
            for pt in valid_pts:
                ax.scatter(pt[0], pt[1], pt[2], color='red', s=100, edgecolors='black', depthshade=False)
                
            ax.set_title("Surface Topology & Computed Critical Points")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        except Exception as e:
            res_str += f"\nPlotting error: {e}"
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, res_str)

        self.canvas.draw()

if __name__ == "__main__":
    app = AdvancedMathUI()
    app.mainloop()