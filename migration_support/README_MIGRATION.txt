Stargazer portable dual-strategy package

Target folder:
  C:\Users\10757\trading\Stargazer

StockDB folder:
  C:\Users\10757\trading\free-stockdb-windows-v0.3.2-more-power\stockdb

First use:
  1. Extract the Stargazer folder to C:\Users\10757\trading.
  2. Run install_runtime.bat once (internet required, installs CPU inference only).
  3. Update data in StockDB.
  4. Run refresh_and_score.bat after the close. First full refresh normally takes 5-15 minutes.
  5. Run Stargazer.exe.

Daily use:
  Update StockDB -> refresh_and_score.bat -> Stargazer.exe.

Green-dip notices are research/human-review signals. They never submit orders.
The UI shows T+1/T+2/T+3 timing, the selected -2%..-8% trigger, reference price,
and whether the point model has positive utility. Minute snapshots are saved locally.

API keys are intentionally empty. Local model scoring and slash commands do not need Kimi.
Fill Kimi/Tavily/push keys in config.yaml only if those optional features are needed.
