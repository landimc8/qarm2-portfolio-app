# Biodiesel Feedstock Hedging Optimizer

A Streamlit app for our QARM II project (HEC Lausanne, Prof. Marc-Aurèle Divernois).

**Live app:** https://biodiesel-hedging-optimizer.streamlit.app/

**Team:** Betty Mikeal, Danela Vlacic, Imen Hassine and Zildete Landim Crausaz

## The idea
 
Our client, **Rotterdam Biofuels SA** (fictional), makes biodiesel. It has to buy raw materials (feedstocks) such as vegetable oils and animal fats, and their prices move a lot.
 
The app answers one question: **which mix of feedstocks keeps the client's costs as stable as possible?**
 
We treat the feedstocks like a portfolio and use the minimum variance method from class (Markowitz).
 
## The client's three rules
 
1. **Spend 100% of the budget**, with no short selling.
2. **No more than 35% in one feedstock**, so the client is not too dependent on one supplier. Tallow counts as one feedstock, so live cattle (LE=F) and Darling (DAR) together stay under 35%.
3. **At least 40% waste-based feedstocks** (UCO and animal fat), for the EU rules on biofuels (RED II).
You can change the last two rules with the sliders in the app. The cap can't go below 25%: with 4 real feedstocks, a lower cap can't add up to 100%.
 
## The feedstocks and their proxies
 
Most real feedstock prices are not free, so we use free Yahoo Finance prices that move in a similar way:
 
| Real feedstock | Proxy we use |
|---|---|
| Soybean oil | ZL=F (soybean oil futures) |
| Rapeseed | ZS=F (US soybean futures) |
| Used cooking oil (UCO) | HO=F (US diesel futures) |
| Tallow (animal fat) | LE=F (live cattle futures) and DAR (Darling Ingredients) |
 
## What the app shows
 
- **The efficient frontier**, with and without the client's rules.
- **Our recommended mix** (minimum variance) and the Sharpe-equivalent mix for comparison.
- **Global minimum variance formula** from the course (no rules, short selling allowed), compared with our mix.
- **Risk contributions**: which feedstock brings the most risk.
- **Cost-at-Risk**: how much the costs could go up in a bad month, in dollars.
- **How much does our mix reduce cost swings?** Our mix compared with buying only one feedstock or an equal split.
- **Why minimum variance, not Sharpe**: the client is a buyer, so stable costs matter more than "return".
- **Stable costs or stable margin?** A second goal for a client that sells at diesel-linked prices (tracking error vs diesel, LN2). Fixed selling price: use our main mix. Selling price that follows diesel: use the stable-margin mix.
- **Two basis-risk checks**: how well our proxies follow the real European prices.
  - Diesel (HO=F) vs Rotterdam gasoil: a good proxy.
  - US soybeans (ZS=F) vs Euronext rapeseed: a weak proxy (correlation about 0.44).
- **The crack spread**: the difference between diesel and crude oil prices.
## Data
 
- Portfolio prices: **Yahoo Finance** (free), downloaded by the app.
- Rotterdam gasoil and Euronext rapeseed: **Investing.com** CSV files in the `data` folder, used only for the checks.
## How to run it on your computer
 
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
 
The app then opens in your browser.
 
## Main files
 
| File | What it does |
|---|---|
| `streamlit_app.py` | The app: sliders, charts, tables and the check boxes |
| `data_loader.py` | Downloads the prices and computes returns |
| `optimizer.py` | Computes the frontier, minimum variance and Sharpe |
| `constraints.py` | The client's rules, including the tallow cap |
| `data/` | The gasoil and rapeseed files for the checks |
 
## Limits
 
- **Proxies are not the real prices.** Rapeseed is the weakest one, so the client should hedge it with Euronext futures.
- **The mix does not lock prices.** To fix a price, the client would also need futures contracts.
- **Prices are in US dollars**, but the client works in euros.
- **The mix is based on past data**, and it is tested on the same data. It should be re-run every month, and a backtest on newer data would make the test stronger.