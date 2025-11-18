"""
Monte Carlo STOIIP Simulator - Streamlit Version
University of Kirkuk - College of Engineering - Petroleum Department

Developed by: Bilal Rabah & Omar Yilmaz
Supervised by: Lec. Mohammed Yashar
Date: November 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import truncnorm, qmc
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="STOIIP Monte Carlo Simulator",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (Refined Tab & Info Box Colors) ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .header-container {
        background: linear-gradient(135deg, #1F4E78 0%, #2c5f8d 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        color: white;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.95rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Refined Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        border-radius: 8px 8px 0 0;
        background-color: rgba(255, 255, 255, 0.7);
        border: 1px solid transparent;
        color: #555; /* Default grey text */
        transition: all 0.2s;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1F4E78 !important; /* Active tab background */
        color: white !important; /* Active tab text */
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
        border-top: 2px solid #ddd;
        margin-top: 3rem;
    }
    
    /* Updated Info Box Styling (White Background) */
    .info-box {
        background-color: #ffffff;
        border-left: 5px solid #1F4E78; /* University Blue Accent */
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #333333;
        border: 1px solid #e0e0e0;
        border-left-width: 5px; /* Ensure left border is thicker */
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'params_df' not in st.session_state:
    st.session_state.params_df = None
if 'statistics' not in st.session_state:
    st.session_state.statistics = None

# Default parameters
DEFAULT_PARAMS = {
    'area': {'mean': 200, 'std': 50, 'min': 100, 'max': 300, 'unit': 'acres', 'label': 'Reservoir Area'},
    'h': {'mean': 100, 'std': 25, 'min': 50, 'max': 150, 'unit': 'ft', 'label': 'Net Pay Thickness'},
    'NTG': {'mean': 0.85, 'std': 0.2, 'min': 0.4, 'max': 1.0, 'unit': 'fraction', 'label': 'Net-to-Gross Ratio'},
    'POR': {'mean': 0.32, 'std': 0.02, 'min': 0.28, 'max': 0.36, 'unit': 'fraction', 'label': 'Porosity'},
    'Swi': {'mean': 0.15, 'std': 0.03, 'min': 0.08, 'max': 0.22, 'unit': 'fraction', 'label': 'Initial Water Saturation'},
    'Boi': {'mean': 0.0024, 'std': 0.0001, 'min': 0.0021, 'max': 0.0026, 'unit': 'RB/STB', 'label': 'Oil Formation Volume Factor'}
}

# --- Helper: Robust Image Finder ---
def find_image_path(keywords):
    """
    Searches the current directory for an image file that matches ANY of the keywords.
    Returns the filename if found, else None.
    """
    try:
        files = os.listdir('.')
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                for k in keywords:
                    if k.lower() in f.lower():
                        return f
    except Exception:
        pass
    return None

# Functions
def get_truncated_normal(mean, std, lower, upper):
    """Generate truncated normal distribution parameters"""
    try:
        if std <= 0:
            raise ValueError("Standard deviation must be positive")
        if lower >= upper:
            raise ValueError("Lower bound must be less than upper bound")

        a = (lower - mean) / std
        b = (upper - mean) / std
        return truncnorm(a, b, loc=mean, scale=std)
    except Exception as e:
        st.error(f"Error in truncated normal distribution: {str(e)}")
        return None

def run_simulation(params, n_samples, seed):
    """Execute Monte Carlo simulation using Latin Hypercube Sampling"""
    try:
        # Validate sample count
        if n_samples < 100:
            raise ValueError("Number of samples must be at least 100")
        if n_samples > 1000000:
            raise ValueError("Number of samples cannot exceed 1,000,000")

        # Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=6, seed=seed)
        sample = sampler.random(n=n_samples)

        # Generate samples for each parameter
        data = {}
        param_order = ['area', 'h', 'NTG', 'POR', 'Swi', 'Boi']

        for i, key in enumerate(param_order):
            p = params[key]
            dist = get_truncated_normal(p['mean'], p['std'], p['min'], p['max'])
            if dist is None:
                raise ValueError(f"Failed to create distribution for {key}")
            data[key] = dist.ppf(sample[:, i])

        params_df = pd.DataFrame(data)

        # Calculate STOIIP
        # Formula: N = (7758 * A * h * NTG * phi * (1-Sw)) / Boi
        stoiip = (
            7758 * params_df['area'] *
            params_df['h'] *
            params_df['NTG'] *
            params_df['POR'] *
            (1 - params_df['Swi']) /
            params_df['Boi']
        ) / 1000000 # Convert to MMbbl

        # Check for invalid values
        if stoiip.isnull().any() or (stoiip <= 0).any():
            raise ValueError("Invalid STOIIP values calculated. Check input parameters.")

        results_df = pd.DataFrame({'STOIIP': stoiip})

        # Calculate statistics
        statistics = {
            'Mean': float(stoiip.mean()),
            'Median': float(stoiip.median()),
            'P10 (Optimistic)': float(np.percentile(stoiip, 90)), # P10 is usually higher for reserves
            'P50 (Most Likely)': float(np.percentile(stoiip, 50)),
            'P90 (Conservative)': float(np.percentile(stoiip, 10)), # P90 is usually lower for reserves
            'Std Deviation': float(stoiip.std()),
            'Minimum': float(stoiip.min()),
            'Maximum': float(stoiip.max()),
            'Coefficient of Variation': float(stoiip.std() / stoiip.mean()),
            'Variance': float(stoiip.var())
        }

        return results_df, params_df, statistics

    except Exception as e:
        st.error(f"Simulation error: {str(e)}")
        raise

def create_histogram(stoiip_data, stats):
    """Create interactive histogram using Plotly"""
    try:
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=stoiip_data,
            nbinsx=50,
            name='STOIIP Distribution',
            marker_color='rgba(52, 152, 219, 0.7)',
            marker_line_color='rgb(41, 128, 185)',
            marker_line_width=1.5
        ))

        fig.add_vline(x=stats['Mean'], line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {stats['Mean']:,.0f}",
                      annotation_position="top right")
        fig.add_vline(x=stats['P50 (Most Likely)'], line_dash="dash", line_color="green",
                      annotation_text=f"P50: {stats['P50 (Most Likely)']:,.0f}",
                      annotation_position="top left")

        fig.update_layout(
            title='STOIIP Distribution (Histogram)',
            xaxis_title='STOIIP (MMbbl)',
            yaxis_title='Frequency',
            template='plotly_white',
            hovermode='x',
            height=500
        )

        return fig
    except Exception as e:
        st.error(f"Error creating histogram: {str(e)}")
        return None

def create_cdf(stoiip_data, stats):
    """Create interactive CDF using Plotly"""
    try:
        sorted_stoiip = np.sort(stoiip_data)
        cdf = np.arange(1, len(sorted_stoiip) + 1) / len(sorted_stoiip)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=sorted_stoiip,
            y=cdf,
            mode='lines',
            name='CDF',
            line=dict(color='rgb(155, 89, 182)', width=3)
        ))

        for p_val, color, name in [
            (stats['P90 (Conservative)'], 'red', 'P90'),
            (stats['P50 (Most Likely)'], 'orange', 'P50'),
            (stats['P10 (Optimistic)'], 'green', 'P10')
        ]:
            fig.add_vline(x=p_val, line_dash="dash", line_color=color,
                          annotation_text=f"{name}: {p_val:,.0f}")

        fig.update_layout(
            title='Cumulative Distribution Function (CDF)',
            xaxis_title='STOIIP (MMbbl)',
            yaxis_title='Cumulative Probability',
            template='plotly_white',
            hovermode='x',
            height=500
        )

        return fig
    except Exception as e:
        st.error(f"Error creating CDF: {str(e)}")
        return None

def create_box_plot(params_df):
    """Create box plot for parameter distributions"""
    try:
        fig = go.Figure()

        params_to_plot = ['area', 'h', 'NTG', 'POR', 'Swi', 'Boi']

        for param in params_to_plot:
            if param in params_df.columns:
                fig.add_trace(go.Box(
                    y=params_df[param],
                    name=DEFAULT_PARAMS[param]['label'],
                    boxmean='sd'
                ))

        fig.update_layout(
            title='Parameter Distributions (Box Plot)',
            yaxis_title='Value',
            template='plotly_white',
            height=500
        )

        return fig
    except Exception as e:
        st.error(f"Error creating box plot: {str(e)}")
        return None

def export_to_excel(results_df, params_df, statistics, params_input):
    """Export results to Excel file"""
    try:
        output = BytesIO()

        # Flatten params for export
        flat_params = {}
        for key, val in params_input.items():
            for subkey, subval in val.items():
                flat_params[f"{key}_{subkey}"] = [subval]

        params_summary = pd.DataFrame(flat_params).T
        params_summary.columns = ["Value"]

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Results sheet
            results_df.to_excel(writer, sheet_name='STOIIP Results', index=False)

            # Statistics sheet
            stats_df = pd.DataFrame([statistics]).T
            stats_df.columns = ['Value']
            stats_df.to_excel(writer, sheet_name='Statistics')

            # Parameters sheet
            params_summary.to_excel(writer, sheet_name='Input Parameters')

            # Sample data sheet
            params_df.to_excel(writer, sheet_name='Sampled Parameters', index=False)

        return output.getvalue()
    except Exception as e:
        # Return None on error so the button doesn't appear/crash
        return None

# --- Header (Original Design maintained with image support) ---
# Using columns inside the styled div concept isn't strictly possible with markdown alone,
# so we use st.columns below the markdown title if we want images, or inject HTML.
# To keep the *exact* look of the original file but add images, we will use columns.

c1, c2, c3 = st.columns([1, 4, 1])

with c1:
    eng_path = find_image_path(['eng'])
    if eng_path: st.image(eng_path, use_container_width=True)

with c2:
    st.markdown("""
        <div class="header-container">
            <div class="header-title">🛢️ STOIIP Monte Carlo Simulator</div>
            <div class="header-subtitle">
                University of Kirkuk | College of Engineering | Petroleum Engineering Department
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.9rem; opacity: 0.9;">
                Developed by: Bilal Rabah & Omar Yilmaz | Supervised by: Lec. Mohammed Yashar
            </div>
        </div>
    """, unsafe_allow_html=True)

with c3:
    # Look for anniversary/other logo
    a_path = find_image_path(['anniversary', '22', 'a.jpg', 'a.png'])
    if a_path: st.image(a_path, use_container_width=True)


# Sidebar
with st.sidebar:
    st.header("⚙️ Simulation Settings")
    st.markdown("---")

    n_samples = st.number_input(
        "Number of Samples",
        min_value=1000,
        max_value=1000000,
        value=100000,
        step=10000,
        help="Number of Monte Carlo iterations (10k-500k recommended)"
    )

    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=9,
        help="For reproducible results"
    )

    st.markdown("---")
    st.subheader("📊 Quick Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Reset", use_container_width=True):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

    with col2:
        if st.button("📖 Guide", use_container_width=True):
            st.info("""
            **Quick Guide:**
            1. Set parameters
            2. Adjust samples & seed
            3. Click 'Run Simulation'
            4. View results
            5. Export data
            
            **P-Values:**
            - P90: Conservative
            - P50: Most likely
            - P10: Optimistic
            """)

    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; font-size: 0.8rem; color: #666;'>
            <p>© 2025 University of Kirkuk</p>
            <p>All Rights Reserved</p>
        </div>
    """, unsafe_allow_html=True)

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Input Parameters",
    "📊 Results Dashboard",
    "📈 Detailed Analysis",
    "📑 Data Tables",
    "🧪 Methodology"
])

# Tab 1: Input Parameters
with tab1:
    st.header("🎯 Reservoir Parameters Configuration")

    # Updated info box with new CSS class
    st.markdown("""
        <div class="info-box">
            <strong>ℹ️ Information:</strong> Configure the statistical distributions for each reservoir parameter.
            Enter mean, standard deviation, minimum, and maximum values for Monte Carlo sampling.
        </div>
    """, unsafe_allow_html=True)

    params_input = {} # Capture current inputs for export
    col1, col2 = st.columns(2)
    param_keys = list(DEFAULT_PARAMS.keys())

    for idx, key in enumerate(param_keys):
        col = col1 if idx % 2 == 0 else col2

        with col:
            with st.expander(f"🔧 {DEFAULT_PARAMS[key]['label']} ({DEFAULT_PARAMS[key]['unit']})", expanded=True):
                params_input[key] = {}

                cols = st.columns(2)

                with cols[0]:
                    params_input[key]['mean'] = st.number_input(
                        "Mean",
                        value=float(DEFAULT_PARAMS[key]['mean']),
                        key=f"{key}_mean",
                        format="%.6f"
                    )

                    params_input[key]['min'] = st.number_input(
                        "Minimum",
                        value=float(DEFAULT_PARAMS[key]['min']),
                        key=f"{key}_min",
                        format="%.6f"
                    )

                with cols[1]:
                    params_input[key]['std'] = st.number_input(
                        "Std Dev",
                        value=float(DEFAULT_PARAMS[key]['std']),
                        key=f"{key}_std",
                        format="%.6f"
                    )

                    params_input[key]['max'] = st.number_input(
                        "Maximum",
                        value=float(DEFAULT_PARAMS[key]['max']),
                        key=f"{key}_max",
                        format="%.6f"
                    )

                params_input[key]['unit'] = DEFAULT_PARAMS[key]['unit']
                params_input[key]['label'] = DEFAULT_PARAMS[key]['label']

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("▶️ Run Simulation", type="primary", use_container_width=True):
            with st.spinner("🔄 Running Monte Carlo simulation..."):
                try:
                    # Validate inputs
                    valid = True
                    for key in params_input:
                        if params_input[key]['std'] <= 0:
                            st.error(f"❌ Error in {params_input[key]['label']}: Std Dev must be positive")
                            valid = False
                        if params_input[key]['min'] >= params_input[key]['max']:
                            st.error(f"❌ Error in {params_input[key]['label']}: Min must be less than Max")
                            valid = False
                        if params_input[key]['mean'] < params_input[key]['min'] or params_input[key]['mean'] > params_input[key]['max']:
                            st.error(f"❌ Error in {params_input[key]['label']}: Mean must be between Min and Max")
                            valid = False

                    if valid:
                        results_df, params_df, statistics = run_simulation(params_input, n_samples, seed)

                        st.session_state.results_df = results_df
                        st.session_state.params_df = params_df
                        st.session_state.statistics = statistics
                        st.session_state.params_input = params_input # Store for export

                        st.success("✅ Simulation completed successfully!")
                        st.balloons()
                        st.info("📊 View results in the 'Results Dashboard' tab")

                except Exception as e:
                    st.error(f"❌ Simulation failed: {str(e)}")

# Tab 2: Results Dashboard
with tab2:
    if st.session_state.results_df is not None:
        st.header("📊 Simulation Results Dashboard")

        stats = st.session_state.statistics
        stoiip = st.session_state.results_df['STOIIP'].values

        st.subheader("🎯 Key Statistics")

        col1, col2, col3, col4 = st.columns(4)

        metrics = [
            ("Mean STOIIP", stats['Mean'], "#667eea", "#764ba2"),
            ("P50 (Most Likely)", stats['P50 (Most Likely)'], "#f093fb", "#f5576c"),
            ("P90 (Conservative)", stats['P90 (Conservative)'], "#4facfe", "#00f2fe"),
            ("P10 (Optimistic)", stats['P10 (Optimistic)'], "#43e97b", "#38f9d7")
        ]

        for col, (label, value, color1, color2) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, {color1} 0%, {color2} 100%);">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value:,.0f}</div>
                        <div class="metric-label">MMbbl</div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            fig = create_histogram(stoiip, stats)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = create_cdf(stoiip, stats)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Complete Statistical Summary")

        stats_df = pd.DataFrame([stats]).T
        stats_df.columns = ['Value (MMbbl)']
        stats_df['Value (MMbbl)'] = stats_df['Value (MMbbl)'].apply(lambda x: f"{x:,.2f}")
        st.dataframe(stats_df, use_container_width=True)

        st.markdown("---")
        st.subheader("💾 Export Results")

        excel_data = export_to_excel(
            st.session_state.results_df,
            st.session_state.params_df,
            stats,
            st.session_state.params_input
        )

        if excel_data:
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_data,
                file_name=f"STOIIP_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    else:
        st.info("👈 Configure parameters and run simulation to view results")

# Tab 3: Detailed Analysis
with tab3:
    if st.session_state.params_df is not None:
        st.header("📈 Detailed Parameter Analysis")

        # 1. Tornado Chart (Added Improvement)
        st.subheader("🌪️ Sensitivity Analysis (Tornado Chart)")
        # Calculate Spearman Correlation
        corr_data = []
        for col in st.session_state.params_df.columns:
            corr = st.session_state.params_df[col].corr(st.session_state.results_df['STOIIP'], method='spearman')
            corr_data.append({'Parameter': DEFAULT_PARAMS[col]['label'], 'Correlation': corr})

        df_corr = pd.DataFrame(corr_data).sort_values('Correlation', key=abs, ascending=True)

        fig_tornado = px.bar(df_corr, x='Correlation', y='Parameter', orientation='h',
                            title="Parameter Impact on STOIIP (Spearman Rank Correlation)",
                            color='Correlation', color_continuous_scale='RdBu_r')
        fig_tornado.update_layout(height=400)
        st.plotly_chart(fig_tornado, use_container_width=True)
        st.info("Longer bars indicate parameters that have a stronger influence on the oil reserves. Red = Positive correlation (Higher param -> Higher oil), Blue = Negative (Higher param -> Lower oil).")

        st.markdown("---")

        # 2. Convergence Plot (Added Improvement)
        st.subheader("📉 Convergence Plot")
        # Calculate running mean for first 5000 samples to show stability
        data_conv = st.session_state.results_df['STOIIP'].values
        limit = min(5000, len(data_conv))
        running_mean = np.cumsum(data_conv[:limit]) / np.arange(1, limit + 1)
        fig_conv = px.line(y=running_mean, title=f"Mean STOIIP Stability (First {limit} samples)", labels={'x': 'Iterations', 'y': 'Mean STOIIP'})
        fig_conv.update_traces(line_color='#1F4E78')
        st.plotly_chart(fig_conv, use_container_width=True)

        st.markdown("---")

        st.subheader("📊 Parameter Distribution Box Plots")
        fig = create_box_plot(st.session_state.params_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔥 Parameter Correlation Heatmap")

        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = st.session_state.params_df.corr()
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                        square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            plt.title('Parameter Correlation Matrix', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Error creating heatmap: {str(e)}")

    else:
        st.info("👈 Run simulation to view detailed analysis")

# Tab 4: Data Tables
with tab4:
    if st.session_state.results_df is not None:
        st.header("📑 Detailed Data Tables")

        tab4_1, tab4_2, tab4_3 = st.tabs([
            "STOIIP Results",
            "Sampled Parameters",
            "Statistics"
        ])

        with tab4_1:
            st.subheader("📊 STOIIP Calculation Results")
            st.dataframe(st.session_state.results_df.describe(), use_container_width=True)
            st.dataframe(st.session_state.results_df.head(100), use_container_width=True)

            csv = st.session_state.results_df.to_csv(index=False)
            st.download_button(
                "📥 Download STOIIP Results (CSV)",
                csv,
                f"STOIIP_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )

        with tab4_2:
            st.subheader("🔧 Sampled Parameter Values")
            st.dataframe(st.session_state.params_df.describe(), use_container_width=True)
            st.dataframe(st.session_state.params_df.head(100), use_container_width=True)

            csv = st.session_state.params_df.to_csv(index=False)
            st.download_button(
                "📥 Download Parameters (CSV)",
                csv,
                f"Parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )

        with tab4_3:
            st.subheader("📈 Statistical Summary")
            stats_df = pd.DataFrame([st.session_state.statistics]).T
            stats_df.columns = ['Value']
            st.dataframe(stats_df, use_container_width=True)
    else:
        st.info("👈 Run simulation to view data tables")

# Tab 5: Methodology (Added Improvement)
with tab5:
    st.markdown("### Methodology")
    st.markdown("The Stock Tank Oil Initially In Place (STOIIP) is calculated using the volumetric equation:")
    st.latex(r"N = \frac{7758 \cdot A \cdot h \cdot NTG \cdot \phi \cdot (1 - S_{wi})}{B_{oi}}")

    st.markdown("### Variable Definitions")
    st.markdown("""
    | Symbol | Parameter | Unit | Description |
    | :---: | :--- | :--- | :--- |
    | **N** | STOIIP | MMbbl | Stock Tank Oil Initially In Place |
    | **7758**| Constant | - | Conversion factor (acre-ft to bbl) |
    | **A** | Area | acres | Reservoir surface area |
    | **h** | Thickness | ft | Net pay thickness |
    | **NTG** | Net-to-Gross | fraction | Ratio of productive rock to total rock |
    | **φ** | Porosity | fraction | Pore volume fraction |
    | **Sw** | Water Saturation | fraction | Fraction of pore volume occupied by water |
    | **Boi** | FVF | RB/STB | Oil Formation Volume Factor |
    """)

    st.markdown("### Monte Carlo Method")
    st.markdown("""
    This simulator uses **Latin Hypercube Sampling (LHS)** to generate random inputs from defined **Truncated Normal Distributions**.
    1. **Input Distributions:** Each parameter is defined by a Mean, Standard Deviation, Min, and Max.
    2. **Sampling:** The algorithm generates `N` random scenarios (iterations).
    3. **Calculation:** The STOIIP formula is applied to every scenario.
    4. **Analysis:** The resulting distribution of STOIIP values gives P10, P50, and P90 estimates.
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        <p style="font-weight: bold; font-size: 1rem;">University of Kirkuk - College of Engineering</p>
        <p>Petroleum Engineering Department</p>
        <p>Developed by: Bilal Rabah & Omar Yilmaz | Supervised by: Lec. Mohammed Yashar</p>
        <p>© 2025 - All Rights Reserved</p>
    </div>
""", unsafe_allow_html=True)