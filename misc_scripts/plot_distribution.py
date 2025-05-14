import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

ted_df = pd.read_csv(os.path.join(os.getcwd(), 'tedpop/dataset/filtered_ted_refurbished.csv'))
    
# Plotting the distribution of views
plt.figure(figsize=(12, 6))
sns.histplot(ted_df['viewCount'], bins=50, kde=True)
plt.title('Distribution of TED Talk Views')
plt.xlabel('Views')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

ted_df.head() if ted_df is not None else "No TED dataset found."