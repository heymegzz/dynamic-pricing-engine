import nbformat as nbf
from pathlib import Path

def create_notebook():
    nb = nbf.v4.new_notebook()

    code_cells = [
        # Cell 1: imports
        """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path
import sys

# Append project root to path to import config
sys.path.append(str(Path.cwd().parent))
import config
import importlib
importlib.reload(config)""",
        
        # Cell 2: load train.csv, print shape and dtypes
        """df = pd.read_csv(config.TRAIN_FILE, encoding='utf-8', encoding_errors='replace')
print(f"Shape: {df.shape}")
print("\\nData Types:")
print(df.dtypes)""",

        # Cell 3: price distribution — log scale histogram using matplotlib
        """df_price = df[df['price'] > 0].copy()
df_price['log_price'] = np.log1p(df_price['price'])

plt.figure(figsize=(10, 6))
plt.hist(df_price['log_price'], bins=50, edgecolor='k')
plt.title('Log Price Distribution')
plt.xlabel('Log Price')
plt.ylabel('Frequency')
plt.show()""",

        # Cell 4: top 10 categories by listing count — horizontal bar chart
        """df['category_main'] = df['category_name'].str.split('/').str[0]
top_10_cats = df['category_main'].value_counts().head(10)

plt.figure(figsize=(10, 6))
top_10_cats.sort_values().plot(kind='barh', edgecolor='k')
plt.title('Top 10 Main Categories by Listing Count')
plt.xlabel('Count')
plt.ylabel('Category')
plt.show()""",

        # Cell 5: price boxplot by top 5 category_main values
        """top_5_cats = df['category_main'].value_counts().head(5).index
df_top_5 = df[df['category_main'].isin(top_5_cats)]

plt.figure(figsize=(12, 8))
sns.boxplot(data=df_top_5, x='price', y='category_main')
plt.title('Price Distribution by Top 5 Main Categories')
plt.xlabel('Price')
plt.ylabel('Category')
plt.xlim(0, 200) # Limit x-axis to avoid extreme outliers skewing the plot
plt.show()""",

        # Cell 6: missing values — count and percentage table
        """missing_stats = pd.DataFrame({
    'Missing Count': df.isnull().sum(),
    'Missing Percentage': (df.isnull().sum() / len(df)) * 100
})
missing_stats = missing_stats[missing_stats['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
missing_stats""",

        # Cell 7: top 20 brands by listing count — bar chart
        """top_20_brands = df['brand_name'].value_counts().head(20)

plt.figure(figsize=(12, 6))
top_20_brands.plot(kind='bar', edgecolor='k')
plt.title('Top 20 Brands by Listing Count')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.show()""",

        # Cell 8: summary stats — df.describe() on numeric columns
        """df.describe()"""
    ]

    nb['cells'] = [nbf.v4.new_code_cell(code) for code in code_cells]

    out_path = Path(__file__).parent / '01_eda.ipynb'
    with open(out_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print(f"Created notebook successfully at {out_path}")

if __name__ == "__main__":
    create_notebook()
