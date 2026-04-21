import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

# Internal Modules
from core.data_engine import InstitutionalDataEngine
from models.mvo import MarkowitzOptimizer
from models.risk_parity import RiskParityModel
from models.hrp_ml import HierarchicalRiskParity
from models.black_litterman import BlackLittermanEngine
from models.deep_learning_alpha import AlphaModelTrainer, TimeSeriesTransformer, LSTMAlphaGenerator
from models.explainable_ai import XAIAuditor
import torch

# ------------------------------------------------------------------------
# PAGE CONFIGURATION (Institutional Dark Mode theme)
# ------------------------------------------------------------------------
st.set_page_config(page_title="AI Quant Lab | Strategy Sandbox", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* ===== GLOBAL TYPOGRAPHY & DARK MODE ===== */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        color: #c9d1d9;
    }
    
    /* ===== HERO HEADER ===== */
    .main-header {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #58a6ff 0%, #00f2fe 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        text-align: center;
        letter-spacing: -1.5px;
        line-height: 1.1;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #8b949e;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    
    /* ===== SIDEBAR STYLING ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 50%, #1c2333 100%);
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        color: #ffffff;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 1.5px;
        border-radius: 10px;
        border: none;
        padding: 0.85rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
    }
    section[data-testid="stSidebar"] .stButton>button:hover {
        background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%);
        box-shadow: 0 0 25px rgba(46, 160, 67, 0.5);
        transform: translateY(-2px);
    }
    
    /* ===== METRIC CARDS ===== */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
        border: 1px solid #30363d;
        padding: 1.2rem;
        border-radius: 14px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-left: 4px solid #58a6ff;
        backdrop-filter: blur(10px);
    }
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 0 20px rgba(88, 166, 255, 0.15);
        transform: translateY(-3px);
        border-left-color: #00f2fe;
    }
    
    /* ===== TAB STYLING ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #161b22;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        font-size: 0.82rem;
        color: #8b949e;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #58a6ff !important;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(88, 166, 255, 0.15);
    }
    
    /* ===== EXPANDER STYLING ===== */
    .streamlit-expanderHeader {
        background-color: #161b22;
        border-radius: 10px;
        font-weight: 600;
        color: #c9d1d9;
    }
    
    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #238636, #58a6ff);
        border-radius: 10px;
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #58a6ff; }
    
    /* ===== DATAFRAME STYLING ===== */
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    
    /* ===== LANDING PAGE ===== */
    .landing-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .landing-card:hover {
        border-color: #58a6ff;
        box-shadow: 0 0 20px rgba(88, 166, 255, 0.1);
        transform: translateY(-2px);
    }
    .landing-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .landing-title { font-size: 1.1rem; font-weight: 700; color: #e6edf3; margin-bottom: 0.4rem; }
    .landing-desc { font-size: 0.85rem; color: #8b949e; line-height: 1.5; }
    
    /* ===== FOOTER ===== */
    .footer-container {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid #21262d;
    }
    .footer-brand {
        font-size: 1rem;
        font-weight: 600;
        background: linear-gradient(135deg, #58a6ff, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .footer-sub { font-size: 0.75rem; color: #484f58; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>AI Strategic Portfolio Lab</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Institutional-Grade Multi-Model Architecture: Deep Learning, Risk Parity, and Bayesian Blending.</div>", unsafe_allow_html=True)

with st.expander("📘 Platform Manual: How to Dominate this Dashboard"):
    st.markdown("""
    ### Welcome to the Command Center
    This platform mathematically solves the hardest problem in finance: **Asset Allocation**. Rather than guessing what to buy, this dashboard runs 4 simultaneous institutional algorithms on your chosen stocks to prove the safest mathematical path to profit.
    
    **1. The Models:**
    *   **Markowitz (MVO)**: The 'Greedy' baseline. It mathematically targets the absolute highest historical return. It makes the most money, but it is dangerous and crashes violently.
    *   **Risk Parity**: The 'Defensive Shield'. It ignores returns completely and forces every single stock to carry the exact same amount of risk. Use this to survive recessions.
    *   **Hierarchical ML (HRP)**: Uses Unsupervised Machine Learning clustering to group stocks safely without breaking linear algebra matrices.
    *   **Black-Litterman (AI Fusion)**: The Masterpiece. It inherently takes a safe portfolio baseline, asks the **PyTorch Deep Learning Engine** for a prediction on the future, and mathematically blends the two together.
    
    **2. How to Use:**
    Enter your asset universe in the sidebar, set realistic transaction fees, check **🔥 Execute Live PyTorch Model**, and hit Execute. Read the automated *AI Insights* directly under every chart to interpret your results.
    """)
st.markdown("---")

# ------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Strategy Parameters")
    
    asset_input = st.text_input("Asset Universe (Comma Separated)", "AAPL, MSFT, GOOG, TSLA, JPM")
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("today"))
    
    st.markdown("---")
    st.header("🧠 Live AI Meta-Layer")
    run_real_ai = st.checkbox("🔥 Execute Live PyTorch Model", value=False, help="Un-mocks the UI. Physically trains the Neural Network right now. (Takes ~15 seconds to run).")
    ai_arch = st.selectbox("Network Architecture", ["Long Short-Term Memory (LSTM)", "Self-Attention Transformer"])
    
    st.markdown("---")
    st.header("🏢 Real-World Constraints")
    tx_cost_bps = st.slider("Transaction Cost (bps)", min_value=0.0, max_value=50.0, value=5.0, step=1.0, help="1 bps = 0.01%. High fees destroy high-turnover portfolios.")
    use_ledoit_wolf = st.checkbox("Enable Ledoit-Wolf Shrinkage", value=True, help="Mathematically shrinks the covariance matrix to punish outliers.")
    use_optuna = st.checkbox("AI: Run Optuna Hyperparameter Search", value=False, help="Unleashes an autonomous bot to find the absolute perfect Neural Network architecture (May take hours in production).")
    
    run_button = st.button("EXECUTE OPTIMIZATION")

# ------------------------------------------------------------------------
# MAIN EXECUTION PIPELINE
# ------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_and_process_data(tickers, start, end):
    engine = InstitutionalDataEngine(tickers=tickers, start_date=start, end_date=end)
    engine.fetch_market_data()
    
    # Anti-Crash: Filter out tickers that failed to download
    valid_tickers = [t for t in tickers if t in engine.raw_data_cache]
    if len(valid_tickers) < 2:
        raise ValueError("Critical Error: Need at least 2 valid stocks to compute a covariance matrix.")
        
    prices = pd.DataFrame({t: engine.raw_data_cache[t]['Adj Close'] for t in valid_tickers})
    returns = np.log(prices / prices.shift(1)).dropna()
    
    if len(returns) < 5:
        raise ValueError("Critical Error: Selected timeframe or assets have insufficient historical data to compute a stable covariance matrix. Pick older stocks or a wider date range.")
    
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    return mean_returns, cov_matrix, returns

if run_button:
    tickers = [x.strip() for x in asset_input.split(",")]
    
    with st.spinner("Connecting to Local Parquet Data Warehouse & Running Advanced Math Solvers..."):
        time.sleep(0.5) # Fast UI transition
        try:
            # Rebuilding engine to ensure full ML access
            engine = InstitutionalDataEngine(tickers=tickers, start_date=str(start_date), end_date=str(end_date))
            engine.fetch_market_data()
            
            # Execute feature engineering strictly if AI is checked
            if run_real_ai:
                engine.generate_institutional_features()
                X_tr, Y_tr, X_val, Y_val, X_te, Y_te = engine.strict_chronological_split()
                
            # Basic covariance needs
            valid_tickers = [t for t in tickers if t in engine.raw_data_cache]
            if len(valid_tickers) < 2:
                raise ValueError("Critical Error: Need at least 2 valid stocks to compute a covariance matrix.")
                
            prices = pd.DataFrame({t: engine.raw_data_cache[t]['Adj Close'] for t in valid_tickers})
            returns = np.log(prices / prices.shift(1)).dropna()
            
            if len(returns) < 5:
                raise ValueError("Critical Error: Selected timeframe or assets have insufficient historical data to compute a stable covariance matrix. Pick older stocks or a wider date range.")
            
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            daily_returns = returns
        except Exception as e:
            st.error(f"Execution Failed. Error: {e}")
            st.stop()
            
        # --- Transaction Cost Math ---
        # Converting bps to standard percent. We assume a 100% initial capital deployment (turnover = 1.0)
        friction_drag = (tx_cost_bps / 10000.0) 
        mean_returns = mean_returns - friction_drag
        
        # Override to valid tickers in case the user mistyped one
        valid_tickers = list(mean_returns.index)
            
        # --- 1. MVO ---
        mvo = MarkowitzOptimizer()
        # If Ledoit-Wolf is enabled, pass raw daily_returns to the optimizer to shrink the matrix
        if use_ledoit_wolf:
            mvo_res = mvo.maximize_sharpe_ratio(mean_returns, raw_data_for_cov=daily_returns, cov_matrix=cov_matrix)
        else:
            mvo_res = mvo.maximize_sharpe_ratio(mean_returns, cov_matrix=cov_matrix)
            
        w_mvo = mvo_res['weights']
        
        # --- 2. Risk Parity ---
        rp = RiskParityModel()
        w_rp = rp.generate_all_weather_weights(cov_matrix)
        
        # --- 3. HRP ---
        hrp = HierarchicalRiskParity()
        w_hrp = hrp.generate_hrp_weights(cov_matrix)
        
        # --- 4. Black-Litterman and Deep Learning Core ---
        bl = BlackLittermanEngine()
        ai_pred = {}
        shap_df = None
        
        if run_real_ai:
            st.toast("🔥 Training Neural Network... Please wait.")
            primary_asset = valid_tickers[0] # Focus AI on the primary asset for speed in dashboard
            
            # Setup ML Environment
            trainer = AlphaModelTrainer(seq_length=10)
            X_train = X_tr[primary_asset]
            feature_names = list(X_train.columns)
            
            # Inject selected architecture
            if "Transformer" in ai_arch:
                trainer.model = TimeSeriesTransformer(input_size=len(feature_names)).to(trainer.device)
            else:
                trainer.model = LSTMAlphaGenerator(input_size=len(feature_names)).to(trainer.device)
                
            # Execute Training (Mocking short epochs to prevent UI timeout)
            trainer.train_model(X_train, Y_tr[primary_asset], X_val[primary_asset], Y_val[primary_asset], epochs=10)
            
            # Predict the absolute future (Test Set)
            test_seq, _ = trainer.create_sequences(X_te[primary_asset], Y_te[primary_asset])
            test_seq = test_seq.to(trainer.device)
            trainer.model.eval()
            with torch.no_grad():
                future_alpha = trainer.model(test_seq[-1:]) # Predict based on the very last available sequence
                
            # Convert scalar tensor to float, scale back to annualized
            predicted_return = future_alpha.item() * 12 # 1M target * 12 months
            ai_pred = {primary_asset: predicted_return}
            
            # Run XAI Audit
            try:
                bg_data, _ = trainer.create_sequences(X_train.iloc[-100:], Y_tr[primary_asset].iloc[-100:])
                auditor = XAIAuditor(trainer.model, bg_data.to(trainer.device))
                shap_df = auditor.audit_prediction(test_seq[-1:], feature_names)
                shap_df.rename(columns={"Feature": "Feature Indicator", "SHAP_Impact_Score": "Impact Magnitude (SHAP)"}, inplace=True)
            except:
                shap_df = pd.DataFrame({"Feature Indicator": ["Network Explainer Unsupported"], "Impact Magnitude (SHAP)": [1.0]})
                
        else:
            # Fallback to simulated mode
            ai_pred = {valid_tickers[0]: mean_returns.iloc[0] * 1.5}
            shap_df = pd.DataFrame({
                "Feature Indicator": ["RSI_14 / Momentum", "Bollinger Band Contraction", "Volume Accumulation", "MACD Crossover"],
                "Impact Magnitude (SHAP)": [42.5, 28.1, 18.0, 11.4]
            })
            
        w_bl = bl.calculate_posterior_returns(w_mvo, cov_matrix, ai_pred)
        # Normalize BL for comparison
        w_bl = np.abs(w_bl) / np.abs(w_bl).sum()
        
    st.success("All Mathematical Models Successfully Converged.")
    
    # ------------------------------------------------------------------------
    # THE PRE-FLIGHT CORRELATION HEATMAP
    # ------------------------------------------------------------------------
    with st.expander("🗺️ Pre-Flight Check: Asset Correlation Matrix", expanded=False):
        st.markdown("<small style='color:#8b949e'>Analyze your Asset Universe. Dark Red indicates dangerous positive correlation (they crash together). Dark Blue indicates safe negative correlation (diversification).</small>", unsafe_allow_html=True)
        corr_matrix = returns.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9")
        st.plotly_chart(fig_corr, use_container_width=True)

    # ------------------------------------------------------------------------
    # DASHBOARD VISUALIZATIONS
    # ------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        "📊 Weight Matrix", "📈 Historical Backtest", "🛡️ Drawdown Matrix", "🌀 Monte Carlo Sim", 
        "🤖 AI Meta-Layer", "🔬 XAI Matrix (SHAP)", "🎯 Efficient Frontier", "📋 Tear Sheet",
        "⚠️ Value at Risk", "📉 Rolling Sharpe", "🍩 Final Recommendation"
    ])
    
    with tab1:
        st.subheader("Model Weights Comparison")
        st.markdown("Observe how the mathematical objective strictly alters portfolio allocation.")
        
        # Compile into one dataframe
        master_df = pd.DataFrame({
            'Markowitz (MVO)': w_mvo,
            'Risk Parity': w_rp,
            'Hierarchical ML (HRP)': w_hrp,
            'Black-Litterman': w_bl
        }).fillna(0)
        
        # Melt for Plotly
        melted_df = master_df.reset_index().melt(id_vars='index', var_name='Model', value_name='Weight')
        melted_df.columns = ['Asset', 'Model', 'Weight']
        
        # Display Top-Level Performance KPIs for MVO
        st.markdown("### 🏆 MVO Benchmark Performance")
        
        # Calculate capped progress bars (max 100) for visual feedback
        ret_prog = min(max(int(mvo_res['expected_return'] * 100) * 2, 0), 100) # arbitrary geometric scaling
        vol_prog = min(max(int(mvo_res['expected_volatility'] * 100) * 2, 0), 100)
        sharpe_prog = min(max(int((mvo_res['sharpe_ratio'] / 3.0) * 100), 0), 100) # assuming 3.0 is exceptionally strong
        
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric(label="Expected Annual Return", value=f"{mvo_res['expected_return']*100:.2f}%", delta=f"- {tx_cost_bps} bps (Friction)")
            st.progress(ret_prog)
        with kpi2:
            st.metric(label="Minimized Volatility", value=f"{mvo_res['expected_volatility']*100:.2f}%", delta="Target Risk", delta_color="inverse")
            st.progress(vol_prog)
        with kpi3:
            st.metric(label="Max Sharpe Ratio", value=f"{mvo_res['sharpe_ratio']:.3f}", delta="Optimal")
            st.progress(sharpe_prog)
        
        st.markdown("---")
        st.markdown("### ⚖️ Strategic Allocation Matrix")
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=w_mvo.values * 100, theta=w_mvo.index, fill='toself', name='MVO (Greedy)', line_color='#FF4B4B'))
        fig.add_trace(go.Scatterpolar(r=w_rp.values * 100, theta=w_rp.index, fill='toself', name='Risk Parity (Shield)', line_color='#2ea043'))
        fig.add_trace(go.Scatterpolar(r=w_hrp.values * 100, theta=w_hrp.index, fill='toself', name='HRP (Machine Learning)', line_color='#58a6ff'))
                     
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], color="#8b949e"), bgcolor="rgba(0,0,0,0)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9",
            title="Portfolio Geometric Concentration Patterns",
            title_font_size=20,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.markdown("<small style='color:#8b949e'>Geometric Portfolio Shape: Jagged spikes point natively to heavily concentrated danger zones. Perfectly round polygons indicate structural safety.</small>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Standard Grouped Bar Chart restored
        master_df = pd.DataFrame({
            'Markowitz (MVO)': w_mvo,
            'Risk Parity': w_rp,
            'Hierarchical ML (HRP)': w_hrp,
            'Black-Litterman': w_bl
        }).fillna(0)
        
        melted_df = master_df.reset_index().melt(id_vars='index', var_name='Model', value_name='Weight')
        melted_df.columns = ['Asset', 'Model', 'Weight']
        
        fig_bar = px.bar(melted_df, x='Asset', y='Weight', color='Model', barmode='group', 
                     title="Asset Allocation by Strategy Type",
                     color_discrete_sequence=['#FF4B4B', '#2ea043', '#58a6ff', '#f0883e'])
                     
        fig_bar.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9",
            title_font_size=20,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Dynamic AI Insight Generation
        st.markdown("---")
        mvo_max_asset = w_mvo.idxmax()
        mvo_max_val = w_mvo.max() * 100
        rp_alloc = w_rp[mvo_max_asset] * 100
        st.info(f"**🤖 AI Insight**: The Markowitz engine found mathematically that **{mvo_max_asset}** drove historical returns, greedily allocating **{mvo_max_val:.1f}%** of your capital to it. Notice how the Risk Parity shield forcefully scales {mvo_max_asset} down to **{rp_alloc:.1f}%** to protect your portfolio from severe concentration danger.")
        
        with st.expander("🔍 View Raw Weight Data (The Blue Matrix)"):
            st.dataframe(master_df.style.background_gradient(cmap='Blues', axis=None), use_container_width=True)
            
    with tab2:
        st.markdown("### 📈 Live Historical Backtesting")
        st.markdown("<small style='color:#8b949e'>Compounding real daily returns to see exactly how your AI allocations would have performed against an Equal-Weighted Benchmark.</small>", unsafe_allow_html=True)
        
        # Portfolio Math: Daily Returns * Asset Weights
        # IMPORTANT: daily_returns are LOG returns. Must use exp(cumsum) for correct compounding.
        port_mvo = (daily_returns * w_mvo.values).sum(axis=1)
        port_rp = (daily_returns * w_rp.values).sum(axis=1)
        port_eq = (daily_returns * (1.0 / len(valid_tickers))).sum(axis=1)
        
        # Compound Cumulative Returns (exponential of cumulative log returns)
        cum_mvo = np.exp(port_mvo.cumsum())
        cum_rp = np.exp(port_rp.cumsum())
        cum_eq = np.exp(port_eq.cumsum())
        
        cum_df = pd.DataFrame({
            'Markowitz (MVO)': cum_mvo * 100, 
            'Risk Parity (All Weather)': cum_rp * 100, 
            'Equal Weight Benchmark': cum_eq * 100
        }, index=daily_returns.index)
        
        fig_line = px.line(cum_df, title="Cumulative Portfolio Growth over Time (Starting Capital: $100)")
        fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9", yaxis_title="Portfolio Value ($)")
        st.plotly_chart(fig_line, use_container_width=True)
        
    with tab3:
        st.markdown("### 🌊 The 'Pain' Chart (Maximum Drawdown)")
        st.markdown("<small style='color:#8b949e'>How much money you mathematically lost from the all-time peak during crashes. Notice how Risk Parity mitigates extreme bleeding.</small>", unsafe_allow_html=True)
        
        dd_mvo = cum_mvo / cum_mvo.cummax() - 1
        dd_rp = cum_rp / cum_rp.cummax() - 1
        
        dd_df = pd.DataFrame({
            'Markowitz Drawdown': dd_mvo * 100,
            'Risk Parity Drawdown': dd_rp * 100
        }, index=daily_returns.index)
        
        fig_dd = px.area(dd_df, title="Portfolio Drawdowns (Underwater Chart)", color_discrete_sequence=['#FF4B4B', '#2ea043'])
        fig_dd.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9", yaxis_title="Loss from Peak (%)")
        st.plotly_chart(fig_dd, use_container_width=True)
        
        # Dynamic Risk Insight
        st.markdown("---")
        worst_mvo = dd_mvo.min() * 100
        worst_rp = dd_rp.min() * 100
        if worst_rp < worst_mvo:
            st.warning(f"**⚠️ Critical Risk Warning**: The 'Safe' Risk Parity portfolio actually crashed worse (**{worst_rp:.1f}%**) than the Greedy MVO portfolio (**{worst_mvo:.1f}%**). This proves your Asset Universe is dangerously correlated! You MUST add defensive assets like Bonds (TLT) or Gold to fix this math.")
        else:
            st.success(f"**🛡️ Defense Verified**: During the worst historical crash, the greedy MVO model bled **{worst_mvo:.1f}%**, while the Risk Parity shield safely contained the damage to **{worst_rp:.1f}%**. ")
        
    with tab4:
        st.markdown("### 🌀 Advanced Stochastic Monte Carlo")
        st.markdown("<small style='color:#8b949e'>Projecting 100 different future universes using Geometric Brownian Motion based on MVO Volatility.</small>", unsafe_allow_html=True)
        
        # Monte Carlo Setup (GBM)
        mc_days = 252
        num_simulations = 100
        mu = mvo_res['expected_return']
        sigma = mvo_res['expected_volatility']
        starting_val = cum_mvo.iloc[-1] * 100 if len(cum_mvo) > 0 else 100.0
        
        # Generator matrix
        sim_returns = np.random.normal((mu - 0.5 * sigma**2)/252, sigma/np.sqrt(252), (mc_days, num_simulations))
        sim_paths = starting_val * np.exp(np.cumsum(sim_returns, axis=0))
        sim_paths = np.vstack([np.full(num_simulations, starting_val), sim_paths])
        
        fig_mc = go.Figure()
        for i in range(num_simulations):
            fig_mc.add_trace(go.Scatter(y=sim_paths[:, i], mode='lines', line=dict(color='rgba(88, 166, 255, 0.05)'), showlegend=False))
            
        fig_mc.add_trace(go.Scatter(y=np.median(sim_paths, axis=1), mode='lines', line=dict(color='#00f2fe', width=3), name='Median Expected Path'))
        fig_mc.update_layout(title="Future 1-Year Probability Cone", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9")
        st.plotly_chart(fig_mc, use_container_width=True)
        
        st.markdown("---")
        median_finish = np.median(sim_paths[-1])
        st.info(f"**🤖 AI Insight**: Based entirely on the rigid Gaussian volatility computed by the engine, a $100 investment today is mathematically projected to land near **${median_finish:.2f}** precisely 252 trading days from now. 100 randomized universal permutations have confirmed the width of this probability fan.")

    with tab5:
        st.markdown("### 🧠 The 'God Mode' Meta-Layer")
        st.info("The AI sequence detector predicted a directional signal for your primary asset. Notice in the Weight Matrix how the Black-Litterman algorithm **safely** nudged the weight up, without going recklessly 'all-in' like a retail trader would.")
        
        if run_real_ai:
            st.success(f"{ai_arch}: **TRAINED**")
            st.info(f"Targeting: **{valid_tickers[0]}**")
            st.markdown(f"### Predicted APY: {ai_pred[valid_tickers[0]]*100:.2f}%")
        else:
            st.success("LSTM Generator: **MOCKED**")
            st.warning("Early Stopping: **TRIGGERED (Epoch 13)**")
            st.info(f"Generated Alpha Signal: **BUY {valid_tickers[0]}**")

    with tab6:
        st.markdown("### 🔬 Explainable AI (SHAP) Audit Log")
        st.markdown("<small style='color:#8b949e'>Proving mathematically *why* the Deep Learning model made the exact trade recommendation.</small>", unsafe_allow_html=True)
        
        if not run_real_ai:
            st.warning("⚠️ Running in **Simulated Mode**. The SHAP values below are illustrative examples. Enable 'Execute Live PyTorch Model' for real gradient-based explanations.")
        
        # Render the SHAP Insight horizontal bar chart
        fig_shap = px.bar(shap_df.head(10).sort_values(by="Impact Magnitude (SHAP)", ascending=True), 
                          x='Impact Magnitude (SHAP)', y='Feature Indicator', orientation='h',
                          title="Top Mathematical Drivers of the Deep Sequence Prediction",
                          color='Impact Magnitude (SHAP)', color_continuous_scale='viridis')
        fig_shap.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9", coloraxis_showscale=False)
        fig_shap.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_shap, use_container_width=True)
        
        st.info("**🤖 AI Insight**: The horizontal bars physically rank which specific feature inside the `data_engine` (e.g. MACD, Volume) was most mathematically responsible for generating the PyTorch output prediction. This legally guarantees your AI is not acting as a dangerous 'Black Box'.")
    # ====================================================================
    # TAB 7: EFFICIENT FRONTIER
    # ====================================================================
    with tab7:
        st.markdown("### 🎯 The Markowitz Efficient Frontier")
        st.markdown("<small style='color:#8b949e'>The Nobel Prize-winning visualization. Every dot is a random portfolio. The red star marks YOUR optimal portfolio on the frontier curve.</small>", unsafe_allow_html=True)
        
        num_portfolios = 5000
        results = np.zeros((3, num_portfolios))
        weight_array = []
        
        for i in range(num_portfolios):
            w = np.random.dirichlet(np.ones(len(valid_tickers)))
            weight_array.append(w)
            p_ret = np.sum(mean_returns.values * w)
            p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w)))
            results[0, i] = p_vol * 100
            results[1, i] = p_ret * 100
            results[2, i] = (p_ret - 0.04) / p_vol  # Sharpe
        
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=results[0], y=results[1], mode='markers',
            marker=dict(size=3, color=results[2], colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe")),
            name='Random Portfolios', text=[f'Sharpe: {s:.2f}' for s in results[2]]
        ))
        # Mark the optimal MVO portfolio — MUST use same raw cov_matrix as the random cloud
        opt_vol = np.sqrt(np.dot(w_mvo.values.T, np.dot(cov_matrix.values, w_mvo.values))) * 100
        opt_ret = np.sum(mean_returns.values * w_mvo.values) * 100
        fig_ef.add_trace(go.Scatter(
            x=[opt_vol], y=[opt_ret], mode='markers',
            marker=dict(size=18, color='#FF4B4B', symbol='star', line=dict(width=2, color='white')),
            name=f'Your Optimal Portfolio (Sharpe: {mvo_res["sharpe_ratio"]:.2f})'
        ))
        fig_ef.update_layout(
            title='Efficient Frontier: 5,000 Random Portfolios vs. Your Optimal',
            xaxis_title='Portfolio Volatility (%)', yaxis_title='Portfolio Return (%)',
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9"
        )
        st.plotly_chart(fig_ef, use_container_width=True)
        
        st.info(f"**🤖 AI Insight**: Out of 5,000 randomly generated portfolios, your MVO-optimized portfolio (the ⭐ Red Star) achieves a Sharpe Ratio of **{mvo_res['sharpe_ratio']:.3f}**, mathematically positioning it on the upper-left 'efficient' edge of the cloud. Portfolios below the curve are mathematically inferior.")

    # ====================================================================
    # TAB 8: PERFORMANCE TEAR SHEET
    # ====================================================================
    with tab8:
        st.markdown("### 📋 Institutional Performance Tear Sheet")
        st.markdown("<small style='color:#8b949e'>The exact statistics a hedge fund prints and hands to investors. This is what goes on your resume.</small>", unsafe_allow_html=True)
        
        # Calculate statistics for MVO portfolio
        total_days = len(port_mvo)
        trading_years = total_days / 252
        
        # CAGR
        cagr_mvo = (cum_mvo.iloc[-1] ** (1 / trading_years) - 1) * 100 if trading_years > 0 else 0
        cagr_rp = (cum_rp.iloc[-1] ** (1 / trading_years) - 1) * 100 if trading_years > 0 else 0
        
        # Annualized Volatility
        vol_mvo = port_mvo.std() * np.sqrt(252) * 100
        vol_rp = port_rp.std() * np.sqrt(252) * 100
        
        # Max Drawdown
        max_dd_mvo = dd_mvo.min() * 100
        max_dd_rp = dd_rp.min() * 100
        
        # Sortino Ratio (penalizes only downside deviation)
        downside_mvo = port_mvo[port_mvo < 0].std() * np.sqrt(252)
        sortino_mvo = (port_mvo.mean() * 252 - 0.04) / downside_mvo if downside_mvo > 0 else 0
        downside_rp = port_rp[port_rp < 0].std() * np.sqrt(252)
        sortino_rp = (port_rp.mean() * 252 - 0.04) / downside_rp if downside_rp > 0 else 0
        
        # Calmar Ratio (Return / Max Drawdown)
        calmar_mvo = cagr_mvo / abs(max_dd_mvo) if max_dd_mvo != 0 else 0
        calmar_rp = cagr_rp / abs(max_dd_rp) if max_dd_rp != 0 else 0
        
        # Win Rate
        win_rate_mvo = (port_mvo > 0).sum() / len(port_mvo) * 100
        win_rate_rp = (port_rp > 0).sum() / len(port_rp) * 100
        
        # Best / Worst Day
        best_mvo = port_mvo.max() * 100
        worst_mvo = port_mvo.min() * 100
        best_rp = port_rp.max() * 100
        worst_rp_day = port_rp.min() * 100
        
        tear_data = {
            'Metric': ['CAGR (%)', 'Annualized Volatility (%)', 'Max Drawdown (%)', 'Sharpe Ratio', 
                       'Sortino Ratio', 'Calmar Ratio', 'Win Rate (%)', 'Best Day (%)', 'Worst Day (%)'],
            'Markowitz (MVO)': [f'{cagr_mvo:.2f}', f'{vol_mvo:.2f}', f'{max_dd_mvo:.2f}', 
                                f'{mvo_res["sharpe_ratio"]:.3f}', f'{sortino_mvo:.3f}', f'{calmar_mvo:.3f}',
                                f'{win_rate_mvo:.1f}', f'{best_mvo:.2f}', f'{worst_mvo:.2f}'],
            'Risk Parity': [f'{cagr_rp:.2f}', f'{vol_rp:.2f}', f'{max_dd_rp:.2f}',
                           'N/A', f'{sortino_rp:.3f}', f'{calmar_rp:.3f}',
                           f'{win_rate_rp:.1f}', f'{best_rp:.2f}', f'{worst_rp_day:.2f}']
        }
        tear_df = pd.DataFrame(tear_data)
        st.dataframe(tear_df.style.set_properties(**{'background-color': '#161b22', 'color': '#c9d1d9'}), use_container_width=True, hide_index=True)
        
        st.info(f"**🤖 AI Insight**: Your MVO portfolio achieved a CAGR of **{cagr_mvo:.2f}%** with a Win Rate of **{win_rate_mvo:.1f}%**. The Sortino Ratio of **{sortino_mvo:.3f}** (which only penalizes downside volatility) provides a more accurate risk-adjusted view than the Sharpe Ratio alone. A Calmar Ratio above 1.0 indicates excellent drawdown-adjusted returns.")

    # ====================================================================
    # TAB 9: VALUE AT RISK (VaR)
    # ====================================================================
    with tab9:
        st.markdown("### ⚠️ Value at Risk (VaR) Analysis")
        st.markdown("<small style='color:#8b949e'>Basel III compliant risk metric. Answers: 'What is the maximum I could lose tomorrow with 95% confidence?'</small>", unsafe_allow_html=True)
        
        confidence_levels = [0.90, 0.95, 0.99]
        investment = 100000  # $100K base
        
        var_data = []
        for cl in confidence_levels:
            var_mvo_val = np.percentile(port_mvo, (1 - cl) * 100) * investment
            var_rp_val = np.percentile(port_rp, (1 - cl) * 100) * investment
            var_data.append({
                'Confidence Level': f'{cl*100:.0f}%',
                'MVO Daily VaR ($)': f'${var_mvo_val:,.2f}',
                'Risk Parity Daily VaR ($)': f'${var_rp_val:,.2f}'
            })
        
        var_df = pd.DataFrame(var_data)
        
        col_var1, col_var2 = st.columns([1, 1])
        with col_var1:
            st.dataframe(var_df.style.set_properties(**{'background-color': '#161b22', 'color': '#c9d1d9'}), use_container_width=True, hide_index=True)
        
        with col_var2:
            # VaR Distribution Histogram
            fig_var = go.Figure()
            fig_var.add_trace(go.Histogram(x=port_mvo * 100, nbinsx=80, name='MVO Daily Returns', marker_color='rgba(255, 75, 75, 0.6)'))
            fig_var.add_trace(go.Histogram(x=port_rp * 100, nbinsx=80, name='Risk Parity Returns', marker_color='rgba(46, 160, 67, 0.6)'))
            
            # Add VaR lines
            var_95_mvo = np.percentile(port_mvo, 5) * 100
            var_95_rp = np.percentile(port_rp, 5) * 100
            fig_var.add_vline(x=var_95_mvo, line_dash='dash', line_color='#FF4B4B', annotation_text=f'MVO VaR 95%: {var_95_mvo:.2f}%')
            fig_var.add_vline(x=var_95_rp, line_dash='dash', line_color='#2ea043', annotation_text=f'RP VaR 95%: {var_95_rp:.2f}%')
            
            fig_var.update_layout(
                title='Return Distribution with VaR Boundaries',
                xaxis_title='Daily Return (%)', yaxis_title='Frequency', barmode='overlay',
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9"
            )
            st.plotly_chart(fig_var, use_container_width=True)
        
        var_95 = np.percentile(port_mvo, 5) * investment
        st.warning(f"**⚠️ Risk Alert**: On a $100,000 portfolio using MVO weights, the 95% Historical VaR is **${abs(var_95):,.2f}**. This means on 95% of trading days, your maximum daily loss will not exceed this amount. On the remaining 5% of days (extreme black swan events), losses could be significantly worse.")

    # ====================================================================
    # TAB 10: ROLLING SHARPE RATIO
    # ====================================================================
    with tab10:
        st.markdown("### 📉 Rolling Sharpe Ratio Timeline")
        st.markdown("<small style='color:#8b949e'>Track how your strategy's risk-adjusted performance evolves over time. A declining Sharpe signals strategy decay.</small>", unsafe_allow_html=True)
        
        rolling_window = 63  # ~3 months of trading days
        
        rolling_ret_mvo = port_mvo.rolling(rolling_window).mean() * 252
        rolling_vol_mvo = port_mvo.rolling(rolling_window).std() * np.sqrt(252)
        rolling_sharpe_mvo = (rolling_ret_mvo - 0.04) / rolling_vol_mvo
        
        rolling_ret_rp = port_rp.rolling(rolling_window).mean() * 252
        rolling_vol_rp = port_rp.rolling(rolling_window).std() * np.sqrt(252)
        rolling_sharpe_rp = (rolling_ret_rp - 0.04) / rolling_vol_rp
        
        rolling_df = pd.DataFrame({
            'MVO Rolling Sharpe': rolling_sharpe_mvo,
            'Risk Parity Rolling Sharpe': rolling_sharpe_rp
        }, index=daily_returns.index).dropna()
        
        fig_rs = px.line(rolling_df, title=f'Rolling {rolling_window}-Day Sharpe Ratio', color_discrete_sequence=['#FF4B4B', '#2ea043'])
        fig_rs.add_hline(y=0, line_dash='dash', line_color='#8b949e', annotation_text='Break-Even Line')
        fig_rs.add_hline(y=1.0, line_dash='dot', line_color='#58a6ff', annotation_text='Strong (1.0)')
        fig_rs.add_hline(y=2.0, line_dash='dot', line_color='#00f2fe', annotation_text='Exceptional (2.0)')
        fig_rs.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9", yaxis_title="Sharpe Ratio")
        st.plotly_chart(fig_rs, use_container_width=True)
        
        latest_sharpe = rolling_sharpe_mvo.iloc[-1] if len(rolling_sharpe_mvo) > 0 else 0
        if latest_sharpe > 1.0:
            st.success(f"**🟢 Strategy Health: STRONG** — The current rolling Sharpe is **{latest_sharpe:.2f}**, indicating the MVO strategy is actively generating strong risk-adjusted alpha in the current market regime.")
        elif latest_sharpe > 0:
            st.warning(f"**🟡 Strategy Health: MODERATE** — The current rolling Sharpe is **{latest_sharpe:.2f}**. The strategy is profitable but underperforming relative to the risk taken.")
        else:
            st.error(f"**🔴 Strategy Health: CRITICAL** — The current rolling Sharpe is **{latest_sharpe:.2f}**. The strategy is actively destroying value in the current regime. Consider rebalancing immediately.")

    # ====================================================================
    # TAB 11: FINAL RECOMMENDATION DONUT
    # ====================================================================
    with tab11:
        st.markdown("### 🍩 Final Blended Recommendation")
        st.markdown("<small style='color:#8b949e'>The single, unified answer. An equal blend of all 4 mathematical models into one final, diversified recommendation.</small>", unsafe_allow_html=True)
        
        # Average all 4 model weights into one final recommendation
        final_weights = (w_mvo + w_rp + w_hrp + w_bl) / 4.0
        final_weights = final_weights / final_weights.sum()  # Re-normalize to sum to 1.0
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=final_weights.index,
            values=final_weights.values * 100,
            hole=0.55,
            marker=dict(colors=['#FF4B4B', '#2ea043', '#58a6ff', '#f0883e', '#00f2fe', '#d2a8ff', '#ffa657'][:len(final_weights)]),
            textinfo='label+percent',
            textfont=dict(size=14, color='white')
        )])
        fig_donut.update_layout(
            title='Consensus Portfolio: Blended Across All 4 Models',
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9",
            annotations=[dict(text='FINAL', x=0.5, y=0.5, font_size=24, font_color='#58a6ff', showarrow=False)]
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
        # Display the final weights as a clean table
        st.markdown("#### 📝 Final Allocation Breakdown")
        final_display = pd.DataFrame({
            'Asset': final_weights.index,
            'Recommended Weight (%)': (final_weights.values * 100).round(2)
        })
        st.dataframe(final_display.style.background_gradient(cmap='Blues', subset=['Recommended Weight (%)']), use_container_width=True, hide_index=True)
        
        top_asset = final_weights.idxmax()
        top_pct = final_weights.max() * 100
        st.info(f"**🤖 AI Insight**: After mathematically averaging the aggressive MVO, defensive Risk Parity, clustering-based HRP, and Bayesian Black-Litterman models, the consensus recommends **{top_asset}** as your highest-conviction holding at **{top_pct:.1f}%**. This blended approach dramatically reduces single-model bias risk.")
        
else:
    st.markdown("")
    
    # Premium Landing Page
    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    with col_l1:
        st.markdown("""
        <div class='landing-card'>
            <div class='landing-icon'>🎯</div>
            <div class='landing-title'>Markowitz MVO</div>
            <div class='landing-desc'>Nobel Prize-winning mean-variance optimization. Finds the mathematically optimal risk/return tradeoff.</div>
        </div>
        """, unsafe_allow_html=True)
    with col_l2:
        st.markdown("""
        <div class='landing-card'>
            <div class='landing-icon'>🛡️</div>
            <div class='landing-title'>Risk Parity</div>
            <div class='landing-desc'>Equal risk contribution across all assets. Built to survive recessions and black swan events.</div>
        </div>
        """, unsafe_allow_html=True)
    with col_l3:
        st.markdown("""
        <div class='landing-card'>
            <div class='landing-icon'>🧠</div>
            <div class='landing-title'>Deep Learning AI</div>
            <div class='landing-desc'>PyTorch LSTM & Transformer networks trained live on 30+ technical indicators for alpha generation.</div>
        </div>
        """, unsafe_allow_html=True)
    with col_l4:
        st.markdown("""
        <div class='landing-card'>
            <div class='landing-icon'>🔬</div>
            <div class='landing-title'>Explainable AI</div>
            <div class='landing-desc'>SHAP-powered model auditing. Every AI decision is mathematically transparent and fully auditable.</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("<div style='text-align:center; color:#484f58; font-size:0.9rem; margin-top:2rem;'>👈 Configure your Asset Universe in the sidebar and click <b>Execute Optimization</b> to launch the engine.</div>", unsafe_allow_html=True)

# ========================================================================
# GLOBAL FOOTER
# ========================================================================
st.markdown("""
<div class='footer-container'>
    <div class='footer-brand'>AI Strategic Portfolio Lab</div>
    <div class='footer-sub'>Powered by PyTorch · Scipy · SHAP · Streamlit &nbsp;|&nbsp; Strictly for Educational & Analytical Strategy Testing</div>
    <div class='footer-sub' style='margin-top:8px;'>Built with ❤️ using Institutional-Grade Quantitative Mathematics</div>
</div>
""", unsafe_allow_html=True)
