# Biodiesel Feedstock Hedging Optimizer

A Streamlit app for our QARM II project (HEC Lausanne, Prof. Marc-Aurèle Divernois).

**Live app:** https://biodiesel-hedging-optimizer.streamlit.app/

**Team:** Betty Mikeal, Danela Vlacic, Imen Hassine and Zildete Landim Crausaz

## The idea
 
Our fictional client, **Rotterdam Biofuels SA**, produces biodiesel using raw materials such as vegetable oils and animal fats.

Fluctuations in commodity prices create uncertainty around the company's production costs. 
 
Our project explores how diversification and quantitative portfolio optimization can help manage financial exposure to these commodity markets.

We construct a portfolio using market-traded instruments as proxies for biodiesel feedstocks and apply Markowitz minimum-variance optimization to identify an allocation with relatively low financial volatility.

The central question is: 

**which mix of feedstocks keeps the client's costs as stable as possible?**
**Which combination of financial market exposures minimizes portfolio volatility while respecting the client's constraints?**
 
Our focus is on reducing financial portfolio volatility through diversification, while taking into account the producer's exposure to feedstock price fluactuations.

## Investment strategy
Our primary strategy is constrained minimum-variance optimization.
Using historical market data, we estimate the returns, volatilities and correlations of five financial instruments.
We then determine portfolio weights that minimize overall variance while satisfying diversification and thematic allocation constraints.
The strategy is based on three principles:

1. Diversification: Combining instruments whose returns do not move identically can reduce overall portfolio risk.

2. Risk minimization: The primary allocation minimizes portfolio variance rather than maximizing expected returns.

3. Constrained allocation: The portfolio must respect limits designed to reflect the client's risk preferences and feedstock-related priorities.

We also compare the minimum-variance allocation with a maximum-Sharpe portfolio to illustrate the financial risk–return trade-off.

## The client's three rules
 
1. **Full allocation**: Portfolio must sum to 100%, with no short selling.
2. **Concentration limit**: No more than 35% in one feedstock, so the client is not too dependent on one supplier. Tallow counts as one feedstock, so live cattle (LE=F) and Darling (DAR) together stay under 35%.
3. **Waste-related allocation**: At least 40% waste-based feedstocks (UCO and animal fat), for the EU rules on biofuels (RED II).

The concentration and waste-related allocation limits can be adjusted using sliders in the application.

With four modeled feedstock categories, the maximimum category weight cannot be set below 25% while maintaining full allocation.
 
## The feedstocks and their proxies
 
Most real feedstock prices are not free, so we use free Yahoo Finance prices that move in a similar way:
 
| Real feedstock | Proxy we use |
|---|---|
| Soybean oil | ZL=F (soybean oil futures) |
| Rapeseed | ZS=F (US soybean futures) |
| Used cooking oil (UCO) | HO=F (US diesel futures) |
| Tallow (animal fat) | LE=F (live cattle futures) and DAR (Darling Ingredients) |

## What the app shows
## Portfolio optimization
- **The efficient frontier**: Comparison of portfolios with and without the client's allocation constraints.
- **Minimum-variance portfolio**: The allocation that minimizes estimated financial portfolio volatility under the selected constraints.
- **Maximum-Sharpe portfolio**: A financial risk-return benchmark.
- **Global minimum variance formula**: Comparison with the analytical unconstrained solution discussed in class.
- **Risk contributions**: Identification of the instruments contributing most to overall portfolio risk.

##Risk analysis
- **Proxy-based Cost-at-Risk**: how much the costs could go up in a bad month, in dollars.
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
 
The app then opens in your browser
 
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
