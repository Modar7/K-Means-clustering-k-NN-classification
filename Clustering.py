
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

# Verify if the salaries were changed
#print(salary_df.loc[16])
#print(salary_df.loc[65])


# Generate a Boxplot
#salary_df['Salary (in USD)'].plot(kind='box')
#plt.show()

# Generate a Histogram plot
#salary_df['Salary (in USD)'].plot(kind='hist')
#plt.show()

# Minimum and maximum salaries
#print('Minimum salary ' + str(salary_df['Salary (in USD)'].min()))
#print('Maximum salary ' + str(salary_df['Salary (in USD)'].max()))

# Convert the salary values to a numpy array
salary_raw = salary_df['Salary (in USD)'].values

# For compatibility with the SciPy implementation
salary_raw = salary_raw.reshape(-1, 1)
salary_raw = salary_raw.astype('float64')




# Import kmeans from SciPy
from scipy.cluster.vq import kmeans, vq
#from sklearn.cluster import KMeans

# Specify the data and the number of clusters to kmeans()
centroids, avg_distance = kmeans(salary_raw, 4)

# Get the groups (clusters) and distances
groups, cdist = vq(salary_raw, centroids)

plt.scatter(salary_raw, np.arange(0,100), c=groups)
plt.xlabel('Salaries in (USD)')
plt.ylabel('Indices')
#splt.show()
































