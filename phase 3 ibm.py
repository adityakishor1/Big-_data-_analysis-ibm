#Data Ingestion:

import ibm_db

# Connect to the IBM Db2 database
conn = ibm_db.connect("DATABASE=mydb;HOSTNAME=myhost;PORT=myport;PROTOCOL=TCPIP;UID=jtv12964;PWD=1uivTqz3vrbARpGU;", "", "")

# Assuming 'data' is your dataset
for row in data:
    sql = "INSERT INTO airline-delay-and-cancellation-data-2009-2018 (column1, column2, ...) VALUES (?, ?, ...)"
    stmt = ibm_db.prepare(conn, sql)
    ibm_db.bind_param(stmt, 1, row['value1'])
    ibm_db.bind_param(stmt, 2, row['value2'])
    # ...

    if ibm_db.execute(stmt):
        print("Row inserted")
    else:
        print("Error inserting row")

ibm_db.close(conn)


#Data Transformation:

import pandas as pd

# Connect to the IBM Db2 database
conn = ibm_db.connect("DATABASE=mydb;HOSTNAME=myhost;PORT=myport;PROTOCOL=TCPIP;UID=jtv12964;PWD=1uivTqz3vrbARpGU;", "", "")

# SQL query to fetch data
sql = "SELECT * FROM your_table"
data = pd.read_sql(sql, conn)

# Perform basic data transformations
data['new_column'] = data['old_column'] * 2

# Initial Analysis:

import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'data' is your dataset
# Summary statistics
summary = data.describe()

# Histogram of a numeric column
data['numeric_column'].hist()
plt.title('Histogram of Numeric Column')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Scatter plot
data.plot.scatter(x='x_column', y='y_column')
plt.title('Scatter Plot')
plt.xlabel('X Column')
plt.ylabel('Y Column')
plt.show()


# Correlation matrix
correlation_matrix = data.corr()

# Heatmap of the correlation matrix
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

# Box plot to visualize distributions
data.boxplot(column=['numeric_column'], by='categorical_column')
plt.title('Box Plot by Category')
plt.xlabel('Category')
plt.ylabel('Numeric Column')
plt.show()


#Machine Learning:

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Assuming you have prepared your dataset with features (X) and target variable (y)
X = data[['feature1', 'feature2', ...]]
y = data['target_variable']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Assuming you have prepared your dataset with features (X) and target labels (y)
X = data[['feature1', 'feature2', ...]]
y = data['target_label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

from sklearn.cluster import KMeans
import seaborn as sns

# Assuming you have prepared your dataset with features (X)
X = data[['feature1', 'feature2']]

# Create and train a K-Means clustering model
model = KMeans(n_clusters=3)
model.fit(X)

# Predict cluster labels
clusters = model.predict(X)

# Visualize the clusters
data['Cluster'] = clusters
sns.scatterplot(data=data, x='feature1', y='feature2', hue='Cluster')
plt.title('K-Means Clustering')
plt.show()
