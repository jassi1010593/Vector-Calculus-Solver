import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="VAYU TARAL - Advanced Vector Calculus Solver",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='text-align: center; color: #00f2ff;'>Advanced Vector Calculus & Analysis Solver</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>VAYU TARAL Computational Mathematics Suite</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("VAYU TARAL 🚀")
st.sidebar.markdown("**Theorem Solver Suite**")

mode = st.sidebar.selectbox(
    "Select Analysis Module:",
    [
        "Green's Theorem (2D Flow & Circulation)",
        "Stokes' Theorem (Surface Curl vs Line Integral)",
        "Gauss Divergence Theorem (3D Flux)",
        "Maxima & Minima of 2D / 3D Surfaces"
    ]
)

st.sidebar.markdown("---")

# ==========================================
# DYNAMIC INPUT & EXECUTION
# ==========================================

x, y, z = sp.symbols('x y z')

if mode == "Green's Theorem (2D Flow & Circulation)":
    st.subheader("🟢 Green's Theorem Solver")
    st.markdown("Relates a line integral around a simple closed curve $C$ to a double integral over the plane region $D$ bounded by $C$.")
    
    col1, col2 = st.columns(2)
    with col1:
        fx_str = st.text_input("Component F_x (i):", "-y")
    with col2:
        fy_str = st.text_input("Component F_y (j):", "x")
        
    if st.button("Solve & Plot Green's Theorem", type="primary"):
        try:
            F_x = sp.sympify(fx_str)
            F_y = sp.sympify(fy_str)
            
            dq_dx = sp.diff(F_y, x)
            dp_dy = sp.diff(F_x, y)
            integrand = dq_dx - dp_dy
            
            st.success("✅ Analytical Solution Computed Successfully!")
            
            col_res1, col_res2 = st.columns([1, 1])
            with col_res1:
                st.markdown("### 📊 Analytical Breakdown")
                st.write(f"**Vector Field F:** $({F_x})\\mathbf{{i}} + ({F_y})\\mathbf{{j}}$")
                st.latex(r"\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} = " + str(integrand))
                
            with col_res2:
                st.markdown("### 💡 Physical Meaning")
                st.info(
                    "This calculates the macroscopic circulation of the fluid field around a closed loop. "
                    f"If the curl component ({integrand}) > 0, there is a net counter-clockwise fluid rotation inside the area."
                )
                
            # Plotting 2D Quiver Field
            st.markdown("### 📉 2D Vector Field Flow & Rotation Visualization")
            fig, ax = plt.subplots(figsize=(8, 6))
            X, Y = np.meshgrid(np.linspace(-3, 3, 15), np.linspace(-3, 3, 15))
            f_x_num = sp.lambdify((x, y), F_x, 'numpy')(X, Y)
            f_y_num = sp.lambdify((x, y), F_y, 'numpy')(X, Y)
            
            if isinstance(f_x_num, (int, float)): f_x_num = np.full_like(X, f_x_num)
            if isinstance(f_y_num, (int, float)): f_y_num = np.full_like(Y, f_y_num)
                
            ax.quiver(X, Y, f_x_num, f_y_num, color='cyan')
            ax.set_title("2D Vector Field Flow")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Input Parsing Error: {e}")

elif mode == "Stokes' Theorem (Surface Curl vs Line Integral)":
    st.subheader("🔵 Stokes' Theorem Solver")
    st.markdown("Relates the surface integral of the curl of a vector field over a surface $S$ to the line integral of the vector field around its boundary curve $C$.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fx_str = st.text_input("Component F_x (i):", "-y")
    with col2:
        fy_str = st.text_input("Component F_y (j):", "x")
    with col3:
        fz_str = st.text_input("Component F_z (k):", "0")
        
    if st.button("Solve & Plot Stokes' Theorem", type="primary"):
        try:
            F_x = sp.sympify(fx_str)
            F_y = sp.sympify(fy_str)
            F_z = sp.sympify(fz_str)
            
            curl_x = sp.diff(F_z, y) - sp.diff(F_y, z)
            curl_y = sp.diff(F_x, z) - sp.diff(F_z, x)
            curl_z = sp.diff(F_y, x) - sp.diff(F_x, y)
            
            st.success("✅ Analytical Solution Computed Successfully!")
            
            col_res1, col_res2 = st.columns([1, 1])
            with col_res1:
                st.markdown("### 📊 Analytical Breakdown")
                st.write(f"**Vector Field F:** $({F_x})\\mathbf{{i}} + ({F_y})\\mathbf{{j}} + ({F_z})\\mathbf{{k}}$")
                st.latex(r"\nabla \times \mathbf{F} = (" + str(curl_x) + r")\mathbf{i} + (" + str(curl_y) + r")\mathbf{j} + (" + str(curl_z) + r")\mathbf{k}")
                
            with col_res2:
                st.markdown("### 💡 Physical Meaning")
                st.info(
                    "Stokes' theorem relates the line integral of a vector field around a closed loop to the flux of its curl over an open surface. "
                    "It measures how much the vector field spins or swirls around the boundary of the surface patch."
                )
                
            # Plotting 3D Quiver Field
            st.markdown("### 📉 3D Vector Field Topology & Curl")
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            X, Y, Z = np.meshgrid(np.linspace(-2, 2, 8), np.linspace(-2, 2, 8), np.linspace(-2, 2, 8))
            
            fx_n = sp.lambdify((x, y, z), F_x, 'numpy')(X, Y, Z)
            fy_n = sp.lambdify((x, y, z), F_y, 'numpy')(X, Y, Z)
            fz_n = sp.lambdify((x, y, z), F_z, 'numpy')(X, Y, Z)
            
            if isinstance(fx_n, (int, float)): fx_n = np.full_like(X, fx_n)
            if isinstance(fy_n, (int, float)): fy_n = np.full_like(Y, fy_n)
            if isinstance(fz_n, (int, float)): fz_n = np.full_like(Z, fz_n)

            ax.quiver(X, Y, Z, fx_n, fy_n, fz_n, length=0.2, normalize=True, color='deepskyblue', alpha=0.6)
            ax.set_title("3D Vector Field Topology")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Input Parsing Error: {e}")

elif mode == "Gauss Divergence Theorem (3D Flux)":
    st.subheader("🟠 Gauss Divergence Theorem Solver")
    st.markdown("Relates the flux of a vector field through a closed surface to the divergence of the field in the volume enclosed by the surface.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fx_str = st.text_input("Component F_x (i):", "x")
    with col2:
        fy_str = st.text_input("Component F_y (j):", "y")
    with col3:
        fz_str = st.text_input("Component F_z (k):", "z")
        
    if st.button("Solve & Plot Gauss Theorem", type="primary"):
        try:
            F_x = sp.sympify(fx_str)
            F_y = sp.sympify(fy_str)
            F_z = sp.sympify(fz_str)
            
            div = sp.diff(F_x, x) + sp.diff(F_y, y) + sp.diff(F_z, z)
            
            st.success("✅ Analytical Solution Computed Successfully!")
            
            col_res1, col_res2 = st.columns([1, 1])
            with col_res1:
                st.markdown("### 📊 Analytical Breakdown")
                st.write(f"**Vector Field F:** $({F_x})\\mathbf{{i}} + ({F_y})\\mathbf{{j}} + ({F_z})\\mathbf{{k}}$")
                st.latex(r"\text{Div }(\mathbf{F}) = " + str(div))
                
            with col_res2:
                st.markdown("### 💡 Physical Meaning")
                st.info(
                    "Divergence measures the net volumetric 'expansion' or 'contraction' of the field from a point.\n\n"
                    "• If $\\text{div } \\mathbf{F} > 0$, the point acts as a Source (fluid expands outward).\n"
                    "• If $\\text{div } \\mathbf{F} < 0$, the point acts as a Sink (fluid compresses inward)."
                )
                
            # Plotting 3D Divergence Grid
            st.markdown("### 📉 3D Fluid Flux Divergence Grid")
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            X, Y, Z = np.meshgrid(np.linspace(-2, 2, 7), np.linspace(-2, 2, 7), np.linspace(-2, 2, 7))
            
            fx_n = sp.lambdify((x, y, z), F_x, 'numpy')(X, Y, Z)
            fy_n = sp.lambdify((x, y, z), F_y, 'numpy')(X, Y, Z)
            fz_n = sp.lambdify((x, y, z), F_z, 'numpy')(X, Y, Z)
            
            if isinstance(fx_n, (int, float)): fx_n = np.full_like(X, fx_n)
            if isinstance(fy_n, (int, float)): fy_n = np.full_like(Y, fy_n)
            if isinstance(fz_n, (int, float)): fz_n = np.full_like(Z, fz_n)

            ax.quiver(X, Y, Z, fx_n, fy_n, fz_n, length=0.3, color='tomato')
            ax.set_title("3D Divergence Field")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Input Parsing Error: {e}")

elif mode == "Maxima & Minima of 2D / 3D Surfaces":
    st.subheader("📈 Maxima & Minima Surface Analysis")
    st.markdown("Finds critical points of a 2D scalar function $f(x,y)$ and classifies them using the Second Derivative Test ($D = AC - B^2$).")
    
    f_str = st.text_input("Function f(x, y):", "x**3 + y**3 - 3*x - 12*y")
    
    if st.button("Find Extremum & Plot Surface", type="primary"):
        try:
            f = sp.sympify(f_str)
            
            fx = sp.diff(f, x)
            fy = sp.diff(f, y)
            critical_points = sp.solve([fx, fy], (x, y), dict=True)
            
            fxx = sp.diff(fx, x)
            fyy = sp.diff(fy, y)
            fxy = sp.diff(fx, y)

            st.success(f"✅ Found {len(critical_points)} critical point candidates!")
            
            st.markdown("### 📊 Critical Points Classification")
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

                    class_str = "Inconclusive"
                    if D > 0 and A > 0:
                        class_str = "🟢 Local Minimum (Valley)"
                    elif D > 0 and A < 0:
                        class_str = "🔴 Local Maximum (Peak)"
                    elif D < 0:
                        class_str = "🟡 Saddle Point"
                        
                    st.write(f"**Point {index+1}:** $({px_val:.2f}, {py_val:.2f})$ $\\rightarrow f = {pz_val:.2f}$ | **Classification:** {class_str}")

            st.markdown("### 💡 Geometrical Meaning")
            st.info("Maxima are peaks, Minima are valleys, and Saddle points act like structural mountain passes where slopes change signs along different axes.")
            
            # Plotting 3D Surface Topology
            st.markdown("### 📉 Surface Topology & Critical Points")
            fig = plt.figure(figsize=(9, 6))
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(np.linspace(-4, 4, 40), np.linspace(-4, 4, 40))
            
            f_num = sp.lambdify((x, y), f, 'numpy')
            Z = f_num(X, Y)
            if isinstance(Z, (int, float)): Z = np.full_like(X, Z)
            
            surf = ax.plot_surface(X, Y, Z, cmap='terrain', edgecolor='none', alpha=0.7)
            
            for pt in valid_pts:
                ax.scatter(pt[0], pt[1], pt[2], color='red', s=120, edgecolors='black', depthshade=False)
                
            ax.set_title("Surface Topology & Critical Points")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            fig.colorbar(surf, shrink=0.5, aspect=5)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Analysis/Plotting Error: {e}")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 14px;'>"
    "© 2026 All rights Reserved Vayu Taral Solver | Advanced Vector Calculus Module"
    "</p>",
    unsafe_allow_html=True
)