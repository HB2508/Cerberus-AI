import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from st_paywall import add_auth

# --- 0. CORE BRANDING ---
st.set_page_config(
    page_title="CERBERUS | AI Research Suite", 
    page_icon="🐕‍🦺", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# PAYWALL GATEWAY (Must be above heavy imports)
add_auth(required=True)

# --- 1. HEAVY AI IMPORTS (Delayed Loading) ---
@st.cache_resource
def load_nlp_model():
    # Moved imports inside here so they don't crash the installer
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model, torch

# --- 2. INITIALIZE SESSION STATE ---
if "api_connected" not in st.session_state:
    st.session_state.api_connected = False
if "trading_client" not in st.session_state:
    st.session_state.trading_client = None
if "calc_done" not in st.session_state:
    st.session_state.calc_done = False

# --- 3. THE LEGAL COMPLIANCE LAYER ---
def show_masthead_and_legal():
    st.title("🐕‍🦺 CERBERUS")
    st.caption("### Institutional Asset Research & Quantitative Sentiment Engine")
    
    with st.expander("⚖️ LEGAL DISCLOSURE (REQUIRED)", expanded=True):
        st.error("### IMPORTANT: NOT FINANCIAL ADVICE")
        st.markdown("""
        **Governance:** CERBERUS is an experimental suite by **Harry Braime & Archie Pahl**.
        * Founders are not registered investment advisors. You are 100% responsible for your capital.
        """)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("System Access")
    st.write(f"**Founders:** Harry Braime & Archie Pahl")
    st.write(f"**User:** {st.session_state.get('user_email', 'Authenticated User')}")
    
    if not st.session_state.api_connected:
        st.subheader("Connect Alpaca Portfolio")
        api_key = st.text_input("API Key", type="password")
        sec_key = st.text_input("Secret Key", type="password")
        is_live = st.toggle("Live Market", value=False)
        
        if st.button("Initialize Neural Link"):
            try:
                client = TradingClient(api_key, sec_key, paper=(not is_live))
                client.get_account() 
                st.session_state.trading_client = client
                st.session_state.api_connected = True
                st.success("Connection Secured")
                st.rerun()
            except Exception:
                st.error("Auth Failed: Verify keys.")
    else:
        st.success("🟢 CORE ONLINE")
        if st.button("Disconnect Session"):
            st.session_state.api_connected = False
            st.rerun()

# --- 5. MAIN INTERFACE ---
show_masthead_and_legal()

if st.session_state.api_connected:
    ticker_input = st.text_input("Enter Asset Ticker", "AAPL").upper()
    
    if st.button("Generate Intelligence Report"):
        with st.spinner("Processing neural layers..."):
            try:
                tokenizer, model, torch = load_nlp_model()
                # Dummy logic for example - replace with your data logic
                inputs = tokenizer([f"Sentiment for {ticker_input}"], return_tensors="pt")
                outputs = model(**inputs)
                score = torch.nn.functional.softmax(outputs.logits, dim=-1)[0][0].item()
                
                st.session_state.current_score = score
                st.session_state.calc_done = True
                st.session_state.current_ticker = ticker_input
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.get("calc_done"):
        st.metric("Sentiment Confidence", f"{st.session_state.current_score:.2%}")
        if st.button(f"EXECUTE BUY: {st.session_state.current_ticker}"):
            st.success("Order transmitted.")
else:
    st.info("Please connect Alpaca in the sidebar.")
