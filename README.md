# Biodiesel Feedstock Hedging Optimizer

A Streamlit app for our QARM II project (HEC Lausanne, Prof. Marc-Aurèle Divernois).

**Live app:** https://biodiesel-hedging-optimizer.streamlit.app/

**Team:** Betty Mikeal, Danela Vlacic, Imen Hassine and Zildete Landim Crausaz

## For you, Rotterdam Biofuels SA
 
In Rotterdam, you produce renewable diesel and biodiesel (a hypothetical client for this project). You purchase four feedstocks to produce your fuel: rapeseed oil, soybean oil, tallow (animal fat), and used cooking oil (UCO). Each month, their prices fluctuate significantly, which also affects your expenses and profit.
 
This tool answers one question: **which mix of feedstocks makes your procurement cost move the least, while following your rules?**
 
Our goal is stable costs. In the financial sector, "stability" is measured by volatility. Because our "returns" are the daily fluctuations in the prices of your feedstocks, lowering portfolio volatility also implies lowering your expenses. We use Markowitz's minimum variance approach.
 
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
 
1. Buy this mix every month (with a $10M budget, about 10,000 t): 35% rapeseed oil, 35% tallow, 21% soybean oil and 9% UCO. Waste share 44%, no feedstock above 35%.
2. Adhere to your monthly Cost-at-Risk, which is approximately $722,000 in a poor month (one out of every twenty).
3. Use a hedge to secure your margin: 50% of your UCO and tallow with fixed-price supplier contracts, and 100% of your crop oils with futures (about 77 soybean oil and 168 rapeseed contracts per month). Your risk for a bad month drops from roughly $722,000 to roughly $163,000.
4. Maintain the same mix and sell ICE gasoil futures at your selling price if it's in line with diesel.
5. Every quarter, update the mix, preferably using your own prices. When fresh prices are downloaded, the precise figures slightly alter.
## Your three rules
 
1. Spend 100% of your budget, with no short selling.
2. Don't rely too much on one supplier by using no more than 35% of one feedstock. Live cattle (LE=F) and Darling (DAR) together remain below 35% since tallow is considered a single feedstock.
3. For the EU's RED II biofuel regulations, at least 40% of feedstocks must be waste-based (UCO and animal fat). The sliders allow you to modify the final two rules. A lower cap cannot sum up to 100% when there are four actual feedstocks. The cap cannot go below 25%. The waste share (35% UCO + 35% tallow) cannot exceed 70% with a 35% cap.
## How to use the tool
 
1. Use the sidebar to set your rules and budget.
2. See "Recommended allocation" for our suggested mix, which explains how to divide your monthly spending. This is your purchasing strategy.
3. Examine the risk contributions to determine which feedstock has the greatest danger.
4. If your selling price is in line with diesel, open "Stable costs or stable margin?"
5. Use the Profit-at-Risk calculator to see your profit, break-even price, and potential profit for a poor month by entering your selling price, tons, and other expenses. The defaults are comparable to the current market ($1,180 per ton, $100 per ton of other expenditures).
6. Open "Hedging: lock your margin" and click "Use our recommended hedge" to see how much risk a hedge removes, how many futures contracts to buy, and how many tonnes stay at market price.
7. Open "What if you change your rules?" to see how your risk changes if you loosen or tighten the cap or the waste minimum.
8. Upload your own prices in the sidebar ("Add your own data") to run the same optimization on your real prices.
9. Re-run the tool every month: it downloads new prices.
## Your feedstocks and our proxies
 
Your real feedstock prices are not free (companies like Argus and Fastmarkets sell them), so we use free Yahoo Finance prices that move in a similar way:
 
| Your feedstock | Proxy we use | How good is it? |
|---|---|---|
| Used cooking oil (UCO) | HO=F (US diesel futures) | Good: correlation 0.89 with Rotterdam gasoil |
| Tallow (animal fat) | LE=F (live cattle futures) and DAR (Darling Ingredients) | No free tallow price to check |
| Soybean oil | ZL=F (soybean oil futures) | The real contract |
| Rapeseed | ZS=F (US soybean futures) | Weak: correlation about 0.44 with Euronext rapeseed |
 
## What the app shows
 
- The efficient frontier, both with and without your regulations. The safest mix that complies with your guidelines is indicated by the red dot, while the safest mix without any rules is indicated by the gray dot. The price of your rules is the space between them. Instead of chasing the top of the curve, where prices are rising more quickly, buyers choose the leftmost point.
- What makes the gray dot so much quieter than ours? It concentrates on the calmest feedstock, which is about two-thirds live cattle. Your 35% cap entails some risk in return for not depending entirely on one source. The calmest way to split your money is never an even split. The more evenly you spread it, the more your costs move.
- Behind the chart: The mix at the two dots, the flat end of the gray line, and each feedstock independently are shown below the graphic. The blend that is getting close to 100% is the flat portion. Darling: twice the risk, but almost the same average price change as soybeans.
- Explore the curves point by point: Examine the curves point by point by moving a slider along the blue or gray line to view the mix, risk, price change for each risk, and rule checks at each location. Buttons leap to the red dot or the purple diamond, which is the Sharpe-equivalent point—the best price change per risk—where the steepest line from zero hits the blue line. Excellent for investors, but not for consumers.
- Our recommended mix (minimum variance) and the Sharpe-equivalent blend for comparison.
- Global minimum variance formula from the course (no rules, short selling allowed), compared with our mix.
- Risk contributions: which feedstock brings the most risk.
- Cost-at-Risk: the amount in dollars that your expenses could increase during a difficult month.
- Why minimal variance rather than Sharpe: since you are the buyer, a bigger "return" equates to higher expenses.
- To what extent does our mix mitigate cost fluctuations? Our mix was compared to purchasing an equal split or a single feedstock.
- Stable margin or stable costs? If your selling price follows diesel, the stable-margin mix (tracking error vs. diesel) calms your margin. The stable-cost mix plus a gasoil futures hedge is still what we advise because the margin was only 5% calmer with actual market prices.
- Your break-even selling price, your expected monthly profit, and the portion of your earnings that each mix could lose in a poor month are all displayed by the Profit-at-Risk calculator. If the market pays less than your expenses, you have three choices: Cut back on manufacturing, sell a better-paid product (UCOME), or lower your costs.
- Hedging: Use our suggested hedge or decide how much of your margin you want to lock (crop oils with futures, UCO and tallow with forward contracts, selling price with gasoil futures). With the hedge, the futures contracts to purchase, and the tons that remain at market value, it displays your Profit-at-Risk.
- Two basis-risk checks: the degree to which our proxies replicate actual European pricing.
- What happens if you alter your regulations? Your daily risk, cost-at-risk, and largest feedstock for further trash minimums and limitations.
- Your own data: add your actual pricing (from Argus or your invoices) to obtain the stable-margin mix and the suggested blend.
- Recommended data sources for the next version of the tool.
- The crack spread: the difference between diesel and crude oil prices.
## Better with your own data
 
Some of the most useful data is not public, but you have it:
 
- Your real purchase prices and supplier quotes: they would replace our proxies, so the basis risk mostly disappears.
- Your sales contracts, whether they are diesel-linked or have a fixed price, tell us exactly which mix is best for you.
- Your yields and volumes (tonnes of feedstock per ton of fuel) link your tons to the budget.
- Your expenses are expressed in euros, which eliminates the difference between dollars and euros. In the sidebar ("Add your own data"), you can already submit your actual prices: Get the template, complete it, and submit it.
## Recommended data sources (next version)
 
| What it improves | Source we recommend | What it replaces or adds |
|---|---|---|
| Your real feedstock prices (UCO, tallow, rapeseed oil, soybean oil) | Argus or SquareCommodities (spot prices) | Replaces our proxies, so most of the basis risk disappears |
| Your selling price (FAME 0°, UCOME) | Argus or SquareCommodities | Replaces diesel (HO=F) in the stable-margin mix, and gives the real selling price for the Profit-at-Risk calculator and your break-even |
| The real mandate levels | SquareCommodities (regulatory data) | Replaces our assumed 40% RED2 minimum |
| Futures for the hedge | ICE (gasoil, FAME 0°), CBOT (soybean oil), Euronext (rapeseed), your suppliers (UCO, tallow) | With your real prices, turns our rough contract numbers in the hedging box into exact hedge ratios |
| Sustainability proofs | Nabisy | Confirms which volumes count as waste-based for RED2 |
| Market fundamentals | EIA, IEA, JODI (oil and diesel), USDA (oilseeds) | Helps you decide when and how much to hedge |
 
**Links:** [Argus](https://www.argusmedia.com/en/commodities/biofuels-and-feedstocks) · [SquareCommodities](https://www.squarecommodities.com) · [ICE](https://www.ice.com) · [CBOT / CME Group](https://www.cmegroup.com) · [Euronext](https://live.euronext.com) · [Nabisy](https://nabisy.ble.de/app/start) · [EIA](https://www.eia.gov) · [IEA](https://www.iea.org) · [JODI](https://www.jodidata.org) · [USDA FAS](https://fas.usda.gov)
 
Prices (Argus and the exchanges) go into the model. Fundamentals (EIA, IEA, JODI, USDA) and regulatory data (SquareCommodities, Nabisy) set the rules and explain why prices move.
 
## Data
 
- Portfolio prices: Yahoo Finance, which the app downloads for free.
- Investing.com CSV files for Rotterdam Gasoil and Euronext Rapeseed are stored in the data folder and are used only for the checks.
- If you submit your own pricing, they are not preserved and is simply used in the "Your own data" section.
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
 
- Your actual prices are not represented by proxies. Since Rapeseed is the weakest, you should use Euronext futures as a hedge.
- The mix does not fix pricing, but it does lessen how much your costs fluctuate. This is the purpose of the hedge, which has its own expenses, such as cash for margin calls, no profit in the event that prices drop, and basis risk (for instance, you purchase rapeseed oil and Euronext rapeseed is a seed).
- A hedge does not produce a margin; rather, it locks one. A hedge only locks in a loss below your break-even price, which is roughly $1,100 per ton with the settings.
- The hedge figures are approximations, assuming that each hedge accurately reflects your actual price and that there is around one ton of feedstock for every ton of biodiesel. The precise hedge ratios would be determined by a broker.
- You pay in euros, but the prices are in US dollars.
- The blend is tested using the same data and is based on historical data. Every month, run it again; a backtest using more recent data would strengthen the test.
- Because diesel (HO=F) is both our UCO proxy and the selling price, the stable-margin mix appears better with our proxies than it actually is. A futures hedge on the selling side better preserves the margin; with real prices, the gain is typically minimal.
- The tool does not predict prices; instead, it evaluates risk.