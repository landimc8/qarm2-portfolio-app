# Rotterdam Biofuels SA — Feedstock Cost-Risk Hedging

**QARM II Project (HEC Lausanne)**

**Live app:** https://biodiesel-hedging-optimizer.streamlit.app/

> **Status:** the app is working. The full documentation (methodology, data provenance, limitations) is being added.

## What it does

An interactive Streamlit tool that builds a feedstock procurement strategy for a biodiesel producer. It minimizes cost volatility while respecting three client rules: a minimum share of waste-based feedstocks (EU RED2), a concentration cap per feedstock, and a procurement budget. It uses constrained mean-variance optimization, with risk budgeting as a second lens.

## Files

- `streamlit_app.py`: the app interface (run this one)
- `data_loader.py`: downloads and cleans prices from Yahoo Finance
- `constraints.py`: the client's rules
- `optimizer.py`: the optimization and risk budgeting math
- `requirements.txt`: Python packages needed

## Run locally

```
git clone https://github.com/landimc8/qarm2-portfolio-app.git
cd qarm2-portfolio-app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Team

Betty Mikeal, Danela Vlacic, Imen Hassine, Zildete Landim Crausaz