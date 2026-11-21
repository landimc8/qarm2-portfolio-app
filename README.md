# Biodiesel Feedstock Hedging Optimizer

A Streamlit app for our QARM II project (HEC Lausanne, Prof. Marc-Aurèle Divernois).

**Live app:** https://biodiesel-hedging-optimizer.streamlit.app/

**Team:** Betty Mikeal, Danela Vlacic, Imen Hassine and Zildete Landim Crausaz

## For you, Rotterdam Biofuels SA
 
You make biodiesel and renewable diesel in Rotterdam (a fictional client for this project). To make your fuel, you buy four feedstocks: used cooking oil (UCO), tallow (animal fat), soybean oil and rapeseed oil. Their prices move a lot every month, and when they move, your costs and your profit move too.
 
This tool answers one question: **which mix of feedstocks makes your procurement cost move the least, while following your rules?**
 
Our goal is stable costs. In finance, "stable" is measured by volatility. Our "returns" are the daily price changes of your feedstocks, so minimizing the portfolio's volatility means minimizing how much your costs move. We use the minimum variance method from the course (Markowitz).
 
## Your needs → our answer
 
| Your need | What the tool gives you |
|---|---|
| Stable feedstock costs | A minimum-variance buying plan |
| Keep your RED2 benefit | At least 40% waste feedstock, checked live (PASS/FAIL) |
| Don't depend on one supplier | At most 35% per feedstock, tallow counted once |
| Know your risk in money | Cost-at-Risk and a Profit-at-Risk calculator |
| Your selling price follows diesel | A stable-margin mix |
| Trust the data | Basis-risk checks on diesel and rapeseed |
 
## Your three rules
 
1. **Spend 100% of your budget**, with no short selling.
2. **No more than 35% in one feedstock**, so you don't depend too much on one supplier. Tallow counts as one feedstock, so live cattle (LE=F) and Darling (DAR) together stay under 35%.
3. **At least 40% waste-based feedstocks** (UCO and animal fat), for the EU rules on biofuels (RED II).
You can change the last two rules with the sliders. The cap can't go below 25%: with 4 real feedstocks, a lower cap can't add up to 100%. With a 35% cap, the waste share can't go above 70% (35% UCO + 35% tallow).
 
## How to use the tool
 
1. Set **your budget** and **your rules** in the sidebar.
2. Read **our recommended mix** in "Recommended allocation": how to split your budget each month. This is your buying plan.
3. Check the **risk contributions**: which feedstock brings the most risk.
4. Open **"Stable costs or stable margin?"** if your selling price follows diesel.
5. Use the **Profit-at-Risk calculator**: enter your selling price, tonnes and other costs to see how much profit a bad month could take.
6. Re-run the tool **every month**: it downloads new prices.
## Your feedstocks and our proxies
 
Your real feedstock prices are not free (companies like Argus and Fastmarkets sell them), so we use free Yahoo Finance prices that move in a similar way:
 
| Your feedstock | Proxy we use | How good is it? |
|---|---|---|
| Used cooking oil (UCO) | HO=F (US diesel futures) | Good: correlation 0.89 with Rotterdam gasoil |
| Tallow (animal fat) | LE=F (live cattle futures) and DAR (Darling Ingredients) | No free tallow price to check |
| Soybean oil | ZL=F (soybean oil futures) | The real contract |
| Rapeseed | ZS=F (US soybean futures) | Weak: correlation about 0.44 with Euronext rapeseed |
 
## What the app shows
 
- **The efficient frontier**, with and without your rules. The gap is the cost of your rules.
- **Our recommended mix** (minimum variance) and the Sharpe-equivalent mix for comparison.
- **Global minimum variance formula** from the course (no rules, short selling allowed), compared with our mix.
- **Risk contributions**: which feedstock brings the most risk.
- **Cost-at-Risk**: how much your costs could go up in a bad month, in dollars.
- **Why minimum variance, not Sharpe**: you are a buyer, so a higher "return" means higher costs.
- **How much does our mix reduce cost swings?** Our mix compared with buying only one feedstock or an equal split.
- **Stable costs or stable margin?** Fixed selling price: use our main mix. Selling price that follows diesel: use the stable-margin mix (tracking error vs diesel, LN2).
- **Profit-at-Risk calculator**: your expected monthly profit, and how much of it a bad month could take with each mix.
- **Two basis-risk checks**: how well our proxies follow the real European prices.
- **The crack spread**: the difference between diesel and crude oil prices.
## Better with your own data
 
Some of the most useful data is not public, but you have it:
 
- **Your real purchase prices and supplier quotes**: they would replace our proxies, so the basis risk mostly disappears.
- **Your sales contracts** (fixed price or diesel-linked): they tell us directly which mix fits you.
- **Your volumes and yields** (tonnes of feedstock per tonne of fuel): they connect the budget to your tonnes.
- **Your costs in euros**: they remove the dollar/euro gap.
## Recommended data sources (next version)
 
| What it improves | Source we recommend | What it replaces or adds |
|---|---|---|
| Your real feedstock prices (UCO, tallow, rapeseed oil, soybean oil) | **Argus** | Replaces our proxies, so most of the basis risk disappears |
| Your selling price (FAME, HVO) | **Argus** | Replaces diesel (HO=F) as the benchmark for the stable-margin mix |
| The real mandate levels | **SquareCommodities** (regulatory data) | Replaces our assumed 40% RED2 minimum |
| Futures for the hedge | **ICE** (gasoil), **CME** (soybean oil), **Euronext** (rapeseed) | With Argus prices, gives the hedge ratio (Lab 2) to lock your prices |
| Sustainability proofs | **Nabisy** | Confirms which volumes count as waste-based for RED2 |
| Market fundamentals | **EIA, IEA, JODI** (oil and diesel), **USDA** (oilseeds) | Helps you decide when and how much to hedge |
 
Prices (Argus and the exchanges) go into the model. Fundamentals (EIA, IEA, JODI, USDA) and regulatory data (SquareCommodities, Nabisy) set the rules and explain why prices move.
 
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
| `constraints.py` | Your rules, including the tallow cap |
| `data/` | The gasoil and rapeseed files for the checks |
 
## Before you decide: limits
 
- **Proxies are not your real prices.** Rapeseed is the weakest one, so you should hedge it with Euronext futures.
- **The mix reduces how much your costs move, but it does not lock prices.** To fix a price, you would also need futures contracts.
- **Prices are in US dollars**, but you pay in euros.
- **The mix is based on past data**, and it is tested on the same data. Re-run it every month; a backtest on newer data would make the test stronger.
- **The tool measures risk; it does not forecast prices.**