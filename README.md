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
| Know your risk in money | Cost-at-Risk and a Profit-at-Risk calculator, with your break-even price |
| Lock your margin | A hedging plan: futures for crop oils, forward contracts for UCO and tallow |
| Your selling price follows diesel | Our mix + a gasoil futures hedge (the stable-margin mix is an option to test) |
| Trust the data | Basis-risk checks on diesel and rapeseed |
 
## Our recommendation in short
 
1. **Buy this mix every month** (with a $10M budget, about 10,000 t): 35% rapeseed oil, 35% tallow, 21% soybean oil and 9% UCO. Waste share 44%, no feedstock above 35%.
2. **Follow your Cost-at-Risk every month**: about $722,000 in a bad month (1 month in 20).
3. **Lock your margin with a hedge**: 100% of your crop oils with futures (about 77 soybean oil and 168 rapeseed contracts a month) and 50% of your UCO and tallow with fixed-price supplier contracts. Your bad-month risk falls from about $722,000 to about $163,000.
4. **If your selling price follows diesel**, keep the same mix and also sell ICE gasoil futures for your selling price.
5. **Update the mix every quarter**, ideally with your own prices.
The exact numbers change a little when new prices are downloaded.
 
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
5. Use the **Profit-at-Risk calculator**: enter your selling price, tonnes and other costs to see your profit, your break-even price, and how much profit a bad month could take. The defaults ($1,180 per tonne, $100 per tonne of other costs) are close to today's market.
6. Open **"Hedging: lock your margin"** and click **"Use our recommended hedge"** to see how much risk a hedge removes, how many futures contracts to buy and how many tonnes stay at market price.
7. Open **"What if you change your rules?"** to see how your risk changes if you loosen or tighten the cap or the waste minimum.
8. Upload **your own prices** in the sidebar ("Add your own data") to run the same optimization on your real prices.
9. Re-run the tool **every month**: it downloads new prices.
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
- **Stable costs or stable margin?** The stable-margin mix (tracking error vs diesel) makes your margin calmer if your selling price follows diesel. Our recommendation stays the stable-cost mix plus a gasoil futures hedge: with real market prices, the margin was only about 5% calmer.
- **Profit-at-Risk calculator**: your expected monthly profit, your break-even selling price, and how much of your profit a bad month could take with each mix. If the market pays less than your costs, it shows your three choices: lower your costs, sell a better-paid product (UCOME), or produce less.
- **Hedging: lock your margin**: choose how much you lock (crop oils with futures, UCO and tallow with forward contracts, selling price with gasoil futures), or use our recommended hedge. It shows your Profit-at-Risk with the hedge, the futures contracts to buy and the tonnes that stay at market price.
- **Two basis-risk checks**: how well our proxies follow the real European prices.
- **What if you change your rules?** Your daily risk, Cost-at-Risk and biggest feedstock for other caps and waste minimums.
- **Your own data**: upload your real prices (Argus or your invoices) and get the recommended mix and the stable-margin mix on them.
- **Recommended data sources** for the next version of the tool.
- **The crack spread**: the difference between diesel and crude oil prices.
## Better with your own data
 
Some of the most useful data is not public, but you have it:
 
- **Your real purchase prices and supplier quotes**: they would replace our proxies, so the basis risk mostly disappears.
- **Your sales contracts** (fixed price or diesel-linked): they tell us directly which mix fits you.
- **Your volumes and yields** (tonnes of feedstock per tonne of fuel): they connect the budget to your tonnes.
- **Your costs in euros**: they remove the dollar/euro gap.
You can already upload your real prices in the sidebar ("Add your own data"): download the template, fill it in, and upload it.
 
## Recommended data sources (next version)
 
| What it improves | Source we recommend | What it replaces or adds |
|---|---|---|
| Your real feedstock prices (UCO, tallow, rapeseed oil, soybean oil) | **Argus** or **SquareCommodities** (spot prices) | Replaces our proxies, so most of the basis risk disappears |
| Your selling price (FAME 0°, UCOME) | **Argus** or **SquareCommodities** | Replaces diesel (HO=F) in the stable-margin mix, and gives the real selling price for the Profit-at-Risk calculator and your break-even |
| The real mandate levels | **SquareCommodities** (regulatory data) | Replaces our assumed 40% RED2 minimum |
| Futures for the hedge | **ICE** (gasoil, FAME 0°), **CBOT** (soybean oil), **Euronext** (rapeseed), your suppliers (UCO, tallow) | With your real prices, turns our rough contract numbers in the hedging box into exact hedge ratios |
| Sustainability proofs | **Nabisy** | Confirms which volumes count as waste-based for RED2 |
| Market fundamentals | **EIA, IEA, JODI** (oil and diesel), **USDA** (oilseeds) | Helps you decide when and how much to hedge |
 
Prices (Argus and the exchanges) go into the model. Fundamentals (EIA, IEA, JODI, USDA) and regulatory data (SquareCommodities, Nabisy) set the rules and explain why prices move.
 
## Data
 
- Portfolio prices: **Yahoo Finance** (free), downloaded by the app.
- Rotterdam gasoil and Euronext rapeseed: **Investing.com** CSV files in the `data` folder, used only for the checks.
- Your own prices, if you upload them, are used only in the "Your own data" box and are not saved.
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
- **The mix reduces how much your costs move, but it does not lock prices.** That is what the hedge is for, and a hedge has its own costs: cash for margin calls, no gain if prices fall, and basis risk (for example Euronext rapeseed is a seed, and you buy rapeseed oil).
- **A hedge locks a margin, it does not create one.** Below your break-even price (about $1,100 per tonne with the defaults), a hedge only locks in a loss.
- **The hedge numbers are estimates**: they assume each hedge follows your real price perfectly and about 1 tonne of feedstock per tonne of biodiesel. A broker would set the exact hedge ratios.
- **Prices are in US dollars**, but you pay in euros.
- **The mix is based on past data**, and it is tested on the same data. Re-run it every month; a backtest on newer data would make the test stronger.
- **With our proxies, the stable-margin mix looks better than it is**, because diesel (HO=F) is both our UCO proxy and the selling price. With real prices, the gain is usually small; a futures hedge on the selling side protects the margin better.
- **The tool measures risk; it does not forecast prices.**