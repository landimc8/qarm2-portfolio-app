# Biodiesel Feedstock Hedging Optimizer

A Streamlit app for our QARM II project (HEC Lausanne, Prof. Marc-Aurèle Divernois).

**Live app:** https://biodiesel-hedging-optimizer.streamlit.app/

**Team:** Betty Mikeal, Danela Vlacic, Imen Hassine and Zildete Landim Crausaz

## For you, Rotterdam Biofuels SA
 
In Rotterdam, you produce renewable diesel and biodiesel (a hypothetical client for this project). You purchase four feedstocks to produce your fuel: rapeseed oil, soybean oil, tallow (animal fat), and used cooking oil (UCO). Each month, their prices fluctuate significantly, which also affects your expenses and profit.
 
This tool answers one question: **which mix of feedstocks makes your procurement cost move the least, while following your rules?**
 
Our goal is stable costs. In the financial sector, "stability" is measured by volatility. Because our "returns" are the daily fluctuations in the prices of your feedstocks, lowering portfolio volatility means your expenses move less. We use Markowitz's minimum variance approach.
 
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
| Stay close to what you already buy | A benchmark mix (your current mix, diesel or a greener target) and a slider between low risk and staying close to it (tracking error) |
| Use your own price views | Black-Litterman: enter a view (e.g. "UCO will rise 20%") and the mix adjusts |
| Know if it works on new data | A monthly backtest, out of sample, compared with an equal split |
| Compare other methods | Minimum CVaR and risk parity (ERC), next to our mix |
| More feedstocks and real European prices | An extended universe of 15 feedstocks, with shrinkage, with a "4 / 15 feedstocks" button in most sections |
| Compare with the commodity market | Tracking error, beta and correlation vs the Bloomberg Commodity Index and three other benchmarks |
 
## Our recommendation in short
 
1. Buy this mix every month (with a $10M budget, about 10,000 t): 35% rapeseed oil, 35% tallow, 21% soybean oil and 9% UCO. Waste share 44%, no feedstock above 35%.
2. Adhere to your monthly Cost-at-Risk, which is approximately $722,000 in a poor month (one out of every twenty).
3. Use a hedge to secure your margin: 50% of your UCO and tallow with fixed-price supplier contracts, and 100% of your crop oils with futures (about 77 soybean oil and 168 rapeseed contracts per month). Your risk for a bad month drops from roughly $722,000 to roughly $163,000.
4. Maintain the same mix and sell ICE gasoil futures at your selling price if it's in line with diesel.
5. Every quarter, update the mix, preferably using your own prices. When fresh prices are downloaded, the precise figures slightly alter.
6. With more feedstocks and real European prices (15 feedstocks), your daily cost risk falls from 0.96% to about 0.68%, and your Cost-at-Risk from about $722,000 to about $514,000. Tested on prices the mix had not seen (backtest), it moves 0.61% a day, against 1.00% for an equal split. A commodity index is a weak hedge for you: it explains only about a third of your cost swings, so hedge with your own feedstock futures.
## Your three rules
 
1. Spend 100% of your budget, with no short selling.
2. Don't rely too much on one supplier by using no more than 35% of one feedstock. Live cattle (LE=F) and Darling (DAR) together remain below 35% since tallow is considered a single feedstock. In the extended universe, feeder cattle (GF=F) is a third tallow proxy and counts in the same cap.
3. For the EU's RED II biofuel regulations, we assume at least 40% of feedstocks are waste-based (UCO and animal fat). The sliders allow you to modify the final two rules. A lower cap cannot sum up to 100% when there are four actual feedstocks. The cap cannot go below 25%. The waste share (35% UCO + 35% tallow) cannot exceed 70% with a 35% cap. With the 15 feedstocks of the extended universe, the cap can go down to 10% and the waste share up to 75% or more.
## How to use the tool
 
1. Use the sidebar to set your rules and budget.
2. See "Recommended allocation" for our suggested mix, which explains how to divide your monthly spending. This is your purchasing strategy. Use the switch "4 feedstocks / Extended universe (15 feedstocks)" to see the same section, and risk budgeting, with 15 feedstocks.
3. Examine the risk contributions to determine which feedstock has the greatest danger.
4. Open "Your benchmark" to stay close to a mix you choose, "Your price views: Black-Litterman" to add your own views, "Backtest" to see how the strategy did on new data, and "Other strategies" to compare with minimum CVaR and risk parity. Each of these boxes has a button to switch between 4 and 15 feedstocks.
5. Use the Profit-at-Risk calculator to see your profit, break-even price, and potential profit for a poor month by entering your selling price, tons, and other expenses. The defaults are comparable to the current market ($1,180 per ton, $100 per ton of other expenditures).
6. Open "Hedging: lock your margin" and click "Use our recommended hedge" to see how much risk a hedge removes, how many futures contracts to buy, and how many tonnes stay at market price.
7. Open "What if you change your rules?" to see how your risk changes if you loosen or tighten the cap or the waste minimum.
8. Upload your own prices in the sidebar ("Add your own data") to run the same optimization on your real prices.
9. Re-run the tool every month: it downloads new prices.
10. Open "Extended universe" to see the 15 feedstocks, the effect of shrinkage, and how your costs compare with a commodity benchmark (choose BCOM, BCOMAG, DBA or GSG).
## Your feedstocks and our proxies
 
Your real feedstock prices are not free (companies like Argus and Fastmarkets sell them), so we use free Yahoo Finance prices that move in a similar way:
 
| Your feedstock | Proxy we use | How good is it? |
|---|---|---|
| Used cooking oil (UCO) | HO=F (US diesel futures) | Good: correlation 0.89 with Rotterdam gasoil |
| Tallow (animal fat) | LE=F (live cattle futures) and DAR (Darling Ingredients) | No free tallow price to check |
| Soybean oil | ZL=F (soybean oil futures) | The real contract |
| Rapeseed | ZS=F (US soybean futures) | Weak: correlation about 0.44 with Euronext rapeseed |
 
### Extended universe: 15 feedstocks
 
In the extended universe, we use real exchange prices where we have them and add more proxies. Prices are weekly (to avoid holiday and contract-roll jumps) and stay in their own currency.
 
| Feedstock | Price used | Source | Waste-based |
|---|---|---|---|
| UCO | ICE gasoil, Rotterdam | Investing.com file | yes |
| UCO | Heating oil (HO=F) | Yahoo | yes |
| Tallow | Live cattle (LE=F), Darling (DAR), feeder cattle (GF=F) | Yahoo | yes (one cap for all three) |
| Pork fat / white grease | Lean hogs (HE=F) | Yahoo | yes |
| Soybean oil | ZL=F | Yahoo | no |
| Soybeans | ZS=F | Yahoo | no |
| Rapeseed | Euronext (EUR) | Investing.com file | no |
| Canola | ICE (CAD) | Investing.com file | no |
| Palm oil | Bursa Malaysia (MYR) | Investing.com file | no (high-ILUC: comparison only) |
| Distillers corn oil | Corn (ZC=F) | Yahoo | no |
| Company proxies | Neste (NESTE.HE), Verbio (VBK.DE), Bunge (BG) | Yahoo | no |
 
## What the app shows
 
- The efficient frontier, both with and without your regulations. The safest mix that complies with your guidelines is indicated by the red dot, while the safest mix without any rules is indicated by the gray dot. The price of your rules is the space between them. Instead of chasing the top of the curve, where prices are rising more quickly, buyers choose the leftmost point. A button shows the same chart with 15 feedstocks: both curves move to the left, and your rules cost less.
- What makes the gray dot so much quieter than ours? It concentrates on the calmest feedstock, which is about two-thirds live cattle. Your 35% cap entails some risk in return for not depending entirely on one source. The calmest way to split your money is never an even split. The more evenly you spread it, the more your costs move.
- Explore the curves point by point: Examine the curves point by point by moving a slider along the blue or gray line to view the mix, risk, price change for each risk, and rule checks at each location. Buttons leap to the red dot or the purple diamond, which is the Sharpe-equivalent point—the best price change per risk—where the steepest line from zero hits the blue line. Excellent for investors, but not for consumers.
- Our recommended mix (minimum variance) and the Sharpe-equivalent blend for comparison. A switch shows the same section, and risk budgeting, with 4 or 15 feedstocks, for both minimum variance and Sharpe-equivalent.
- Global minimum variance formula from the course (no rules, short selling allowed), compared with our mix.
- Risk contributions: which feedstock brings the most risk.
- Cost-at-Risk: the amount in dollars that your expenses could increase during a difficult month.
- Why minimal variance rather than Sharpe: since you are the buyer, a bigger "return" equates to higher expenses. A button shows the same comparison with 15 feedstocks.
- Your benchmark (tracking error): pick a benchmark mix (your current mix, diesel or a greener target) and a slider moves you between the lowest risk and staying close to it.
- Your price views (Black-Litterman): enter how much you expect a feedstock price to move and how sure you are; the mix moves away from the feedstocks you expect to get more expensive.
- Backtest: the mix is rebuilt every month on past data only and tested on the next month, compared with an equal split. With 15 feedstocks (past two years of weekly prices), it also compares the mix with and without shrinkage.
- Other strategies: minimum CVaR (the mix that limits the worst days) and risk parity (each feedstock brings the same risk), next to our mix.
- Extended universe: 15 feedstocks, Ledoit-Wolf shrinkage, and your costs compared with a commodity benchmark (tracking error, beta, correlation).
- Your break-even selling price, your expected monthly profit, and the portion of your earnings that each mix could lose in a poor month are all displayed by the Profit-at-Risk calculator. If the market pays less than your expenses, you have three choices: Cut back on manufacturing, sell a better-paid product (UCOME), or lower your costs. A third column shows the Profit-at-Risk with the 15-feedstock mix.
- Hedging: Use our suggested hedge or decide how much of your margin you want to lock (crop oils with futures, UCO and tallow with forward contracts, selling price with gasoil futures). With the hedge, the futures contracts to purchase, and the tons that remain at market value, it displays your Profit-at-Risk. The hedge stays on the 4 feedstocks you actually buy.
- Two basis-risk checks: the degree to which our proxies replicate actual European pricing. Both explain why the extended universe uses the real Rotterdam gasoil and Euronext rapeseed prices.
- What happens if you alter your regulations? Your daily risk, cost-at-risk, and largest feedstock for further waste minimums and limitations. With 15 feedstocks, a 70% waste minimum has a Cost-at-Risk of about $622,000, against $910,000 with 4.
- Your own data: add your actual pricing (from Argus or your invoices) to obtain the stable-margin mix and the suggested blend.
- Recommended data sources for the next version of the tool.
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
| Real US prices of yellow grease, tallow, white grease and corn oil | USDA AMS Market News (free) | Replaces the cattle, hog and corn proxies; free history only since end-2022 |
| Chinese rapeseed oil (ZCE) and European rapeseed oil (CBOT, Argus) | TradingView (paid plan) or the exchanges | More real vegetable-oil prices for the extended universe |
 
**Links:** [Argus](https://www.argusmedia.com/en/commodities/biofuels-and-feedstocks) · [SquareCommodities](https://www.squarecommodities.com) · [ICE](https://www.ice.com) · [CBOT / CME Group](https://www.cmegroup.com) · [Euronext](https://live.euronext.com) · [Nabisy](https://nabisy.ble.de/app/start) · [EIA](https://www.eia.gov) · [IEA](https://www.iea.org) · [JODI](https://www.jodidata.org) · [USDA FAS](https://fas.usda.gov) · [USDA AMS Market News](https://mymarketnews.ams.usda.gov)
 
Prices (Argus and the exchanges) go into the model. Fundamentals (EIA, IEA, JODI, USDA) and regulatory data (SquareCommodities, Nabisy) set the rules and explain why prices move.
 
## Data
 
- Portfolio prices: Yahoo Finance, which the app downloads for free.
- Investing.com CSV files (10 years, daily) are stored in the data folder: Rotterdam gasoil and Euronext rapeseed (used for the checks and the extended universe), ICE canola and Bursa Malaysia palm oil (extended universe), and the Bloomberg Commodity and Bloomberg Agriculture indexes (benchmarks). When a Bloomberg file is missing, the app downloads the index from Yahoo Finance instead.
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
| `data/` | The gasoil and rapeseed files for the checks, the canola and palm oil files for the extended universe, and the Bloomberg index files for the benchmarks |
 
## Before you decide: limits
 
- Proxies are not your real prices. Since Rapeseed is the weakest, you should use Euronext futures as a hedge.
- The mix does not fix pricing, but it does lessen how much your costs fluctuate. This is the purpose of the hedge, which has its own expenses, such as cash for margin calls, no profit in the event that prices drop, and basis risk (for instance, you purchase rapeseed oil and Euronext rapeseed is a seed).
- A hedge does not produce a margin; rather, it locks one. A hedge only locks in a loss below your break-even price, which is roughly $1,100 per ton with the settings.
- The hedge figures are approximations, assuming that each hedge accurately reflects your actual price and that there is around one ton of feedstock for every ton of biodiesel. The precise hedge ratios would be determined by a broker.
- You pay in euros, but the prices are in US dollars.
- The blend is tested using the same data and is based on historical data. Every month, run it again; the backtest box tests it on data the mix has not seen.
- Because diesel (HO=F) is both our UCO proxy and the selling price, the stable-margin mix appears better with our proxies than it actually is. A futures hedge on the selling side better preserves the margin; with real prices, the gain is typically minimal.
- The extended universe mixes real prices with proxies (cattle, hogs, corn) and three company share prices (Neste, Verbio, Bunge), which also move with each company's profits and the stock market. Part of the lower risk with 15 feedstocks comes from these softer proxies.
- In the extended universe, prices stay in their own currency (EUR, CAD, MYR, USD), so the currency effect is not included. Palm oil is high-ILUC: a Dutch buyer could use almost none.
- The Sharpe-equivalent mix with 15 feedstocks depends on average past price changes, which are noisy: it is concentrated and can change a lot with new data. This is one more reason we recommend minimum variance.
- The tool does not predict prices; instead, it evaluates risk.