# Walmart Sales Predictive Analytics

Forecast future sales trends using ML regression models.

## Dataset
- 45 Walmart stores, 6435 records (2010–2012)
- Columns: Store, Date, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment

## Run
pip install pandas matplotlib scikit-learn
python predictive_analysis.py

## Output
- Forecast: 2013 → $1.957B, 2014 → $1.813B, 2015 → $1.668B
- 7-panel dashboard saved as walmart_sales_forecast.png

## Models Used
- Linear Regression (year-based forecasting)
- Multi-Feature Regression (all economic features)
