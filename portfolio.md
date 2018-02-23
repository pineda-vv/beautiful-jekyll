---
layout: page
bigimg: /img/cape_lookout.jpg
---
## Data Science Portfolio

#### **All projects:**
* Python was used for all projects unless otherwise noted.
* Primary machine learning algorithms and tools used were from the sci-kit learn library unless otherwise noted.
* For the individual projects, I sourced and subsequently cleaned all the data that were used in the analysis.
* All projects followed the CRISP-DM methodology.
* Cross-validation was generally used to evaluate any model. This was usually done splitting the data 70/30 randomly or in case of the time series analysis, holding out the most recent time points of the data series.

##### **A short time series analysis of currency exchange**
* **Tools used** - Python, Matplotlib, Statsmodels
* **Primary algorithm** - Autoregressive Integrated Moving Average (ARIMA) model.
* **Summarry** - I took two completely unrelated currency dataset - the Philippine Peso/USD exchange
rate and the Bitcoin Price - and analyzed their fluctuations over time. I built an ARIMA model for each data set and validated each model's forecasting performance. Details can be found [here](https://github.com/pineda-vv/bitcoin_timeseries)
![BTC Forecast](img/confidence.png){:class="img-responsive"}
##### **Recipe recommender - Implicit and Explicit Ratings**
* **Tools used** - Python, Beautiful Soup, Selenium, MongoDB, AWS EC2, PySpark, t-SNE, Latent Dirichlet Allocation, Non-negative Matrix Factorization, Alternating Least Squares (ALS), NLP
* **Summary** - Using data I scraped from two popular recipe websites, I used Spark's Alternating Least Squares (ALS) algorithm to build two recipe recommender systems. In the first website, I used the explicit ratings that users left for each unique recipe. With data from the second site site, I derived implicit ratings by measuring sentiment from the comments left by users for each recipe page. More details can be found [here](https://github.com/pineda-vv/allrecipe_recommender) (explicit ratings) and [here](https://github.com/pineda-vv/Data-Science-Projects/tree/master/recipe_project) (implicit ratings).
![3D LDA gif](img/animated_lda.gif){:class="img-responsive"}
