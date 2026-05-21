"""
 Sales — Predictive Analytics
Assignment: Predictive Modeling to Forecast Future Trends
Dataset: sales_data.csv (45 stores, 6435 records, 2010–2012)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Remove this line if you want interactive window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────
# 1. LOAD & PREPROCESS DATA
# ──────────────────────────────────────────────────────
df = pd.read_csv('sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Year']  = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ──────────────────────────────────────────────────────
# 2. AGGREGATE DATA
# ──────────────────────────────────────────────────────
yearly       = df.groupby('Year')['Weekly_Sales'].sum().reset_index()
monthly_total= df.groupby(df['Date'].dt.to_period('M'))['Weekly_Sales'].sum()
holiday_avg  = df.groupby('Holiday_Flag')['Weekly_Sales'].mean()
store_total  = df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False)
top5         = store_total.head(5)
seasonal     = df.groupby('Month')['Weekly_Sales'].mean()

print("\nYearly Sales:")
print(yearly.to_string(index=False))

# ──────────────────────────────────────────────────────
# 3. MODEL 1 — LINEAR REGRESSION (Year → Total Sales)
# ──────────────────────────────────────────────────────
X = yearly[['Year']]
y = yearly['Weekly_Sales']

model = LinearRegression()
model.fit(X, y)

future = pd.DataFrame({'Year': [2013, 2014, 2015]})
preds  = model.predict(future)
r2     = r2_score(y, model.predict(X))

print(f"\nLinear Regression R²: {r2:.4f}")
print("\nFuture Predictions:")
for yr, p in zip([2013, 2014, 2015], preds):
    print(f"  {yr}: ${p:,.0f}  (${p/1e9:.3f}B)")

# ──────────────────────────────────────────────────────
# 4. MODEL 2 — MULTI-FEATURE REGRESSION
# ──────────────────────────────────────────────────────
features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag']
dfm = df[features + ['Weekly_Sales']].dropna()

mf_model = LinearRegression()
mf_model.fit(dfm[features], dfm['Weekly_Sales'])

mae = mean_absolute_error(dfm['Weekly_Sales'], mf_model.predict(dfm[features]))
r2m = r2_score(dfm['Weekly_Sales'], mf_model.predict(dfm[features]))

print(f"\nMulti-feature R²: {r2m:.4f}")
print(f"Mean Absolute Error: ${mae:,.0f}")

print("\nFeature Coefficients:")
for feat, coef in zip(features, mf_model.coef_):
    print(f"  {feat:15s}: {coef:+.2f}")

# ──────────────────────────────────────────────────────
# 5. VISUALIZATION DASHBOARD
# ──────────────────────────────────────────────────────
BG     = '#0d1117'
CARD   = '#161b22'
ACCENT = '#00d4aa'
ACCENT2= '#ff6b6b'
ACCENT3= '#ffd166'
ACCENT4= '#a78bfa'
TEXT   = '#e6edf3'
MUTED  = '#8b949e'

fig = plt.figure(figsize=(18, 14), facecolor=BG)
fig.suptitle('Walmart Sales — Predictive Analytics Dashboard',
             fontsize=22, fontweight='bold', color=TEXT, y=0.98)
gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
              left=0.06, right=0.97, top=0.93, bottom=0.06)

def style_ax(ax):
    ax.tick_params(colors=TEXT)
    ax.grid(alpha=0.15, color=MUTED)
    for sp in ax.spines.values():
        sp.set_color(MUTED); sp.set_alpha(0.2)

# Plot 1 – Yearly + Forecast
ax1 = fig.add_subplot(gs[0, :2]); ax1.set_facecolor(CARD)
years_all = list(yearly['Year']) + [2013, 2014, 2015]
sales_all = list(yearly['Weekly_Sales']) + list(preds)
ax1.bar(years_all, [s/1e9 for s in sales_all],
        color=[ACCENT]*3 + [ACCENT3]*3, width=0.5, zorder=3)
reg_x = np.linspace(2010, 2015, 100)
ax1.plot(reg_x, [v/1e9 for v in model.predict(pd.DataFrame({'Year': reg_x}))],
         color=ACCENT2, linewidth=2, linestyle='--')
ax1.axvline(x=2012.5, color=MUTED, linestyle=':', alpha=0.5)
ax1.text(2012.7, max(sales_all)/1e9*0.85, '← FORECAST', color=ACCENT3, fontsize=9)
p1 = mpatches.Patch(color=ACCENT,  label='Actual Sales')
p2 = mpatches.Patch(color=ACCENT3, label='Forecast')
ax1.legend(handles=[p1, p2], facecolor=CARD, labelcolor=TEXT, fontsize=9)
ax1.set_title('Yearly Sales & Forecast (2013–2015)', color=TEXT, fontsize=12, pad=8)
ax1.set_ylabel('Sales (Billion $)', color=MUTED, fontsize=10)
style_ax(ax1)

# Plot 2 – Metrics card
ax2 = fig.add_subplot(gs[0, 2]); ax2.set_facecolor(CARD); ax2.axis('off')
ax2.text(0.5, 0.95, 'Model Metrics', ha='center', va='top',
         transform=ax2.transAxes, color=TEXT, fontsize=13, fontweight='bold')
metrics = [('Linear R²', f'{r2:.3f}', ACCENT),
           ('Multi-feat R²', f'{r2m:.3f}', ACCENT4),
           ('MAE', f'${mae/1000:.0f}K', ACCENT2),
           ('Stores', '45', ACCENT3), ('Records', '6,435', TEXT)]
for i, (lbl, val, col) in enumerate(metrics):
    yp = 0.78 - i*0.15
    ax2.text(0.1, yp, lbl, transform=ax2.transAxes, color=MUTED, fontsize=9)
    ax2.text(0.9, yp, val, transform=ax2.transAxes, color=col,
             fontsize=11, fontweight='bold', ha='right')

# Plot 3 – Monthly Trend
ax3 = fig.add_subplot(gs[1, :2]); ax3.set_facecolor(CARD)
xi = np.arange(len(monthly_total))
ax3.fill_between(xi, [v/1e6 for v in monthly_total.values], alpha=0.25, color=ACCENT4)
ax3.plot(xi, [v/1e6 for v in monthly_total.values], color=ACCENT4, linewidth=2)
ticks = [i for i, p in enumerate(monthly_total.index) if p.month == 1]
ax3.set_xticks(ticks)
ax3.set_xticklabels([str(monthly_total.index[i].year) for i in ticks], color=TEXT)
ax3.set_title('Monthly Sales Trend (2010–2012)', color=TEXT, fontsize=12, pad=8)
ax3.set_ylabel('Sales (Million $)', color=MUTED, fontsize=10)
style_ax(ax3)

# Plot 4 – Seasonal
ax4 = fig.add_subplot(gs[1, 2]); ax4.set_facecolor(CARD)
sc = [ACCENT2 if v == seasonal.max() else ACCENT if v >= seasonal.quantile(0.7)
      else ACCENT4 for v in seasonal.values]
ax4.bar(range(1, 13), seasonal.values/1e6, color=sc, width=0.7)
ax4.set_xticks(range(1, 13))
ax4.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'],
                    fontsize=8, color=TEXT)
ax4.set_title('Avg Sales by Month\n(Seasonal Pattern)', color=TEXT, fontsize=11, pad=8)
ax4.set_ylabel('Avg Sales (M$)', color=MUTED, fontsize=9)
style_ax(ax4)

# Plot 5 – Top Stores
ax5 = fig.add_subplot(gs[2, 0]); ax5.set_facecolor(CARD)
ax5.barh(range(5), top5.values/1e9, color=ACCENT, height=0.5)
ax5.set_yticks(range(5))
ax5.set_yticklabels([f'Store {s}' for s in top5.index], color=TEXT, fontsize=9)
ax5.set_title('Top 5 Stores by Revenue', color=TEXT, fontsize=11, pad=8)
ax5.set_xlabel('Total Sales (B$)', color=MUTED, fontsize=9)
style_ax(ax5)

# Plot 6 – Holiday Impact
ax6 = fig.add_subplot(gs[2, 1]); ax6.set_facecolor(CARD)
hvals = [holiday_avg[0]/1e6, holiday_avg[1]/1e6]
ax6.bar(['Non-Holiday', 'Holiday'], hvals, color=[ACCENT4, ACCENT2], width=0.5)
diff = (holiday_avg[1]-holiday_avg[0])/holiday_avg[0]*100
ax6.text(1, hvals[1]+0.01, f'+{diff:.1f}%', ha='center', color=ACCENT2, fontsize=12, fontweight='bold')
ax6.set_title('Holiday vs Non-Holiday\nAvg Weekly Sales', color=TEXT, fontsize=11, pad=8)
ax6.set_ylabel('Avg Sales (M$)', color=MUTED, fontsize=9)
style_ax(ax6)

# Plot 7 – Correlation
ax7 = fig.add_subplot(gs[2, 2]); ax7.set_facecolor(CARD)
cf = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
cv = [df[f].corr(df['Weekly_Sales']) for f in cf]
ax7.barh(cf, cv, color=[ACCENT if v > 0 else ACCENT2 for v in cv], height=0.5)
ax7.axvline(0, color=MUTED, alpha=0.5, linewidth=1)
ax7.set_title('Feature Correlation\nwith Weekly Sales', color=TEXT, fontsize=11, pad=8)
ax7.set_xlabel('Correlation', color=MUTED, fontsize=9)
style_ax(ax7)

plt.savefig('sales_forecast.png', dpi=150,
            bbox_inches='tight', facecolor=BG, edgecolor='none')
print("\n✅ Dashboard saved as sales_forecast.png")







