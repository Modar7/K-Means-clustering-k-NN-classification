# Import the necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





# Create a dataset
# For generating the names we will use a Python library called Faker.
# For generating salaries, we will use the good old numpy
plt.style.use('ggplot')

# Import Faker
# Faker is a Python package that generates fake data for you
from faker import Faker
fake = Faker()

# To ensure the results are reproducible
Faker.seed(4321)

names_list = []

fake = Faker()
for _ in range(100):
  names_list.append(fake.name())

# To ensure the results are reproducible
np.random.seed(7)

salaries = []
for _ in range(100):
    salary = np.random.randint(1000,2500)
    salaries.append(salary)


# We will merge them in a pandas DataFrame.
# Create pandas DataFrame
salary_df = pd.DataFrame(
    {'Person': names_list,
     'Salary (in USD)': salaries
    })

# Print a subsection of the DataFrame
#print(salary_df.head())

salary_df.at[16, 'Salary (in USD)'] = 23
salary_df.at[65, 'Salary (in USD)'] = 17




# First assign all the instances to
salary_df['class'] = 0

# Manually edit the labels for the anomalies
salary_df.at[16, 'class'] = 1
salary_df.at[65, 'class'] = 1

# Veirfy
print(salary_df.loc[16])

salary_df.head()


# Importing KNN module from PyOD
from pyod.models.knn import KNN


# Segregate the salary values and the class labels
X = salary_df['Salary (in USD)'].values.reshape(-1,1)
y = salary_df['class'].values

# Train kNN detector
clf = KNN(contamination=0.02, n_neighbors=5)
clf.fit(X)

# Get the prediction labels of the training data
y_train_pred = clf.labels_

# Outlier scores
y_train_scores = clf.decision_scores_


# Import the utility function for model evaluation
from pyod.utils import evaluate_print

# Evaluate on the training data
evaluate_print('KNN', y, y_train_scores)

# A salary of $37 (an anomaly right?)
X_test = np.array([[37.]])


# Check what the model predicts on the given test data point
clf.predict(X_test)

# A salary of $1256
X_test_abnormal = np.array([[1256.]])

# Predict
clf.predict(X_test_abnormal)


























