import numpy as np
#Getting the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sns.set_style('whitegrid')
sns.set_palette('pastel')

data = pd.read_csv('RFFAnalysisData.csv')

print(data.head())

#Generating Qucik Insights
#Columns
print(data.columns)

#Checking Null Values
print("Null Values")
print(data.isna().sum())

#Dropping NUll Vlaues
data.dropna(inplace=True)

print("\nNull Values After Cleaning")
print(data.isna().sum())

#Checking more info
print(data.info())

#Stastical Exploration
#Since data only contains one numerical value memeber_number and such value
#Cannot give proper insights it is no much imp
print(data.describe(include='all').to_string())
#On sats analysis the data is skewed with mean being greater than median might need to
#normalize or rescale


print("\nDistribution Analysis")
#Distribution Analysis of Frequent Buyers
item_counts = data['Customer_Id'].value_counts().sort_values(ascending=False)
print(item_counts.head().to_string())
#Top 5 Selling Items
print("\nTop 5 Buyers \n")
print(item_counts.head(5).to_string())
#Least 5 Selling Items
print("\nLeast 5 Buyers\n")
print(item_counts.tail(5).to_string())



#To get distribution analysis of date. First I am chaning the date column which is stirng
#to object and then extracting only the date time. Since the timestamp considers
#tmakes all the orders of same date as unique because of milisecond
data['OrderDateYear'] = pd.to_datetime(data['OrderDate']).dt.date
#Distribution Analysis of Ddate
item_counts = data['OrderDateYear'].value_counts().sort_values(ascending=False)
print(item_counts.head().to_string())
#Top 5 Selling Items
print("\nTop 5 Most Selled Date \n")
print(item_counts.head(5).to_string())
#Least 5 Selling Items
print("\nLeast 5 Most Selled Date\n")
print(item_counts.tail(5).to_string())


print("Data Type Changing====")
#Changing Data Types of Colums
data['Customer_Id']=data['Customer_Id'].astype(int)
data['Order_Id']=data['Order_Id'].astype(int)
data['OrderDate'] = pd.to_datetime(data['OrderDate'])
print(data.dtypes)


print("\nRemoving Duplicates\n")
#Removing Duplicates
duplicates = data.duplicated()
print("Number of Duplicates")
print(duplicates.sum())
#Since in this data there are not much information, for example there is no quanitty and
#time stamp for date so we donot know exactly if the two transaction are from same time
#Or different time , hence we are considering it as duplicates
print("After Removing Duplication")
data.drop_duplicates(inplace=True)
print(data.duplicated().sum())



#Plotting Vlaue Counts
data['OrderDateYear'].value_counts().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title('Transaction Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Transaction')
plt.tight_layout()
plt.show()


#Plotting Vlaue Counts Customers
data['Customer_Id'].value_counts().sort_values(ascending=False).head(5).plot(kind='bar')
plt.title('Top 5 Customers')
plt.xlabel('Customers')
plt.ylabel('')
plt.tight_layout()
plt.show()



import datetime
#extracting only date removing timestamp
#Creating a snapshot data as the date to set as threshold to calculate recency
#for this we are selecting a maximum data + 1 as snapshot data
#Also note that we are using only the date not time stamp
snapshotDate = data['OrderDateYear'].max() + datetime.timedelta(days =1)

rfmData = data.groupby(['Customer_Id']).agg({
    'OrderDateYear':lambda x: (snapshotDate - x.max()).days, #recency
    'Order_Id':'count', #Frequency
    'Total':'sum',#Monetary
})

#Renaming the columns
rfmData.columns = ['Recency', 'Frequency', 'Monetary']

print('After RFM and Column\n')
print(rfmData.head())



 # Interpreting the Results
 #
 #    Recency: Lower values indicate more recent purchases.
 #    Frequency: Higher values indicate more frequent purchases.
 #    Monetary: Higher values indicate that the customer has spent more money.
#
#Making a copy of dataframe
rfm = rfmData.copy()
 #Ranking the Data
# Recency: Lower values are better, so invert the ranking
rfm['R_rank'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])

# Frequency: Higher values are better
data_adjusted = rfm['Frequency'] + np.random.rand(len(rfm)) * 0.001  # Adding a tiny noise
rfm['F_rank'] = pd.qcut(data_adjusted, 5, labels=[1, 2, 3, 4, 5])

# Monetary: Higher values are better
rfm['M_rank'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])


print(rfm.head())
# Preview the ranked RFM scores
print(rfm[['R_rank', 'F_rank', 'M_rank']])

def rfm_segment(row):
    if row['R_rank'] >= 4 and row['F_rank'] >= 4 and row['M_rank'] >= 4:
        return 'Best Customers'
    elif row['F_rank'] >= 4:
        return 'Loyal Customers'
    elif row['M_rank'] >= 4 and row['R_rank'] <= 2:
        return 'At Risk'
    elif row['R_rank'] >= 4 and row['F_rank'] >= 2:
        return 'Potential Loyalists'
    elif row['R_rank'] == 5 and row['F_rank'] == 1:
        return 'New Customers'
    elif row['R_rank'] == 3 and row['F_rank'] >= 2:
        return 'Need Attention'
    elif row['R_rank'] <= 2 and row['F_rank'] <= 2 and row['M_rank'] <= 2:
        return 'Hibernating'
    else:
        return 'Other'

# Apply the segmentation
rfm['Segment'] = rfm.apply(rfm_segment, axis=1)

print(rfm.head().to_string())

#Visualizing

sns.countplot(x='Segment', data=rfm,order=['Best Customers', 'Potential Loyalists', 'At Risk', 'Need Attention', 'Other'])
plt.title('Customer Segments Distribution')
plt.xlabel('Segment')
plt.ylabel('Count')
plt.show()


# Scatter plot for R_rank vs F_rank, colored by segment
plt.figure(figsize=(10, 6))
sns.scatterplot(x='R_rank', y='F_rank', hue='Segment', style='M_rank', data=rfm, palette='Set2', s=100)
plt.title('Customer Segments: R_rank vs F_rank')
plt.xlabel('R_rank')
plt.ylabel('F_rank')
plt.legend(title='Customer Segment')
plt.show()

# Creating a pivot table to show the average R, F, M ranks per segment
pivot = rfm.groupby('Segment').mean()

# Plotting heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, cmap='Blues',fmt='.2f')
plt.title('Average R, F, M Ranks by Customer Segment')
plt.show()
#
#
#
#
# # Sample DataFrame with RFM scores and CLV
# # Assume 'CLV' is the Customer Lifetime Value column and 'RFM_Score' is a combined score
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='RFM_Score', y='CLV', size='Monetary', hue='Monetary', data=rfm, palette='coolwarm', sizes=(20, 200))
# plt.title('Customer Lifetime Value vs RFM Score')
# plt.xlabel('RFM Score')
# plt.ylabel('Customer Lifetime Value')
# plt.legend(title='Monetary Contribution', loc='upper left')
# plt.show()
#
#





#Going Back and Clusturing The Data
#Using StatndardScaler to Scale the data
#Lets check the histogram to see if the data is actually skewed

print(rfmData.head().to_string())
#Checking distribution analysis
sns.histplot(rfmData['Recency'])
plt.xlabel('Range of recency')
plt.ylabel('Count')
plt.show()

#I am setting the limit manually as the data is very skewed
sns.histplot(rfmData['Monetary'])
plt.xlabel('Range of monetary')
plt.ylabel('Count')
plt.xlim(0,10000)
plt.ylim(0,750)
plt.show()

print(rfmData['Frequency'].max())
print(rfmData['Frequency'].min())

print(rfmData['Frequency'].value_counts())
print(rfmData['Frequency'].describe())
sns.histplot(rfmData['Frequency'])
plt.xlim(1,980)
plt.ylim(500,2500)
plt.xlabel('Range of frequency')
plt.ylabel('Count')
plt.show()

# The data has a high skew, with most values concentrated between 1 and 2 (25%, 50%, and 75% percentiles are low),
# while the maximum value (max) is 980, indicating a few extreme outliers.

# Create a distribution plot (histogram)
plt.figure(figsize=(10, 6))
sns.histplot(data=rfm, x='Frequency', bins=50, kde=True)
# Set log scale for better visualization of skewed data
plt.xscale('log')
# Add labels and title
plt.title('Distribution of Frequency (Log Scale)', fontsize=14)
plt.xlabel('Frequency (Log Scale)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()



scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfmData)

# Convert scaled data back into a DataFrame
rfm_scaled_df = pd.DataFrame(rfmData, columns=rfmData.columns)
print(rfm_scaled_df.describe())

# Improve Algorithm Performance: Some machine learning algorithms (e.g., k-means clustering,) are sensitive to the scale of the data.
# If features are on different scales (e.g., one feature ranges from 1 to 10,000, while another ranges from 0 to 1),
# the algorithm may give more importance to larger magnitude features.

# Finding the optimal number of clusters
sse = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42,n_init=10)
    kmeans.fit(rfm_scaled_df)
    sse.append(kmeans.inertia_)

# Plot the Elbow method graph
plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal K')
plt.show()

# Apply KMeans
#Optimal Cluster 3
kmeans = KMeans(n_clusters=3, random_state=42)
rfmData['Cluster'] = kmeans.fit_predict(rfm_scaled_df)

# Add cluster labels to the original RFM table
rfmData['Cluster'] = kmeans.labels_

# Analyzing the mean RFM values for each cluster
print(rfmData.groupby('Cluster').mean())

# Plot the clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(data=rfmData, x='Recency', y='Monetary', hue='Cluster', palette='Set1', s=100)
plt.title('RFM Clusters')
plt.show()
