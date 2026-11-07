# LinkedIn launch post

Image: [`docs/assets/linkedin_poster.png`](assets/linkedin_poster.png) (1440 x 1800, `make poster`).
Numbers from `paper/tables/macros.tex`.

---

**A rice-yield model can look 21% better than it really is, just because of how it was tested.**

Machine-learning studies that forecast crop yields from satellite, weather and price data usually answer one question: is the forecast accurate? Two harder questions are rarely asked. Was the evaluation free of leakage? And do the explanations (SHAP and similar) reflect the real mechanisms, or just tell a convincing story?

You can't check explanations against real data, because nobody knows the true cause of each season's yield. So I built a world where the truth *is* known: a process-based simulation of main-season rice in 200 districts of Northeast Thailand (Isan) over 18 seasons. Drought, flood, heat, disease, salinity, prices, competing crops and pest outbreaks all shape the yields, and the true contribution of each one is recorded next to the data.

What came out of it:

🌾 **Leakage hides failures.** Random cross-validation understated the error by 10–21% and hid seasons where the linear model failed badly. Post hoc, I traced those failures to leakage *inside* the tuning loop, from leave-one-out selection of the penalty.
🛰️ **Satellites carry the signal.** They were the only feature group whose removal hurt LightGBM (+25% RMSE), and their value showed up only from October.
🔍 **Explanations are only partly right.** SHAP recovered drought well (r = 0.84) but attributed heat stress to soil moisture and folded the technology trend into prices. A fragile model gave explanations that looked confident and were wrong.
📉 **Accuracy:** LightGBM cut the error by 42% against climatology (27.4 vs 47.1 kg/rai), and the gain held in 10 independent simulated worlds.
📏 **Uncertainty:** conformal 80% intervals covered 78.3% of outcomes.

Everything is open, and one command (`make all`) rebuilds every number, table and figure:
📄 Paper (read online): https://thanatchon36.github.io/isan-rice-yield-benchmark/paper/
💻 Code: https://github.com/thanatchon36/isan-rice-yield-benchmark
🤗 Dataset with the true yield components: https://huggingface.co/datasets/thanatchon36/isan-rice-yield-benchmark
🔖 DOI: 10.5281/zenodo.24036674

All data are synthetic. This is a benchmark for testing methods, not a statement about Thai rice yields, and there is a Google Earth Engine pathway for plugging in real MODIS, CHIRPS and ERA5-Land data. If you work on crop forecasting, agricultural insurance or explainable AI, I'd love feedback on the design.

*Views are my own.*

#MachineLearning #ExplainableAI #RemoteSensing #Agriculture #FoodSecurity #Thailand #DataScience #OpenScience
