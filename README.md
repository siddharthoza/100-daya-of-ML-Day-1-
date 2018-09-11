# 100-daya-of-ML-Day-1-
6 most frequently used libraries for Machine Learning
Getting started with Machine learning
Complete guide on how to import different files in python
1. Import Different types of libraries
i. Requests
I have used request library a lot while doing my web scraping project. Basically Request will allow you to send HTTP/1.1 requests using Python. With it, you can add content like headers, form data, multipart files, and parameters via simple Python libraries. It also allows you to access the response data of Python in the same way. For that first you need to install Requests package (code in anaconda prompt: pip install requests) and then you can import the request library

import requests
url = 'https://www.worldofwatches.com/mens-watches'
response=requests.get(url)
This is bascially extract the html page of the given url
ii. Beautiful Soup
This is the best library to parse the html/XML file or data for the beginners. In order to use this library first install the ps4 package and then import the beautiful soup library.

## (pip install ps4)
from bs4 import BeautifulSoup # Used to parse the HTML content of web pages
soup = BeautifulSoup(response.text, 'html.parser')
result = soup.find_all(class_= 'filter-options-content')
This link is the complete guide for scarping web pages using beautiful soup https://www.digitalocean.com/community/tutorials/how-to-scrape-web-pages-with-beautiful-soup-and-python-3
iii. Numpy
Numpy is the core library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays. Numpy Arrays: A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers. The number of dimensions is the rank of the array; the shape of an array is a tuple of integers giving the size of the array along each dimension.

import numpy as np
​
a = np.array([1, 2, 3])   # Create a rank 1 (1 dimensional) array
print(type(a))            # Prints "<class 'numpy.ndarray'>"
print(a.shape)            # Prints "(3,)"
print(a[0], a[1], a[2])   # Prints "1 2 3"
a[0] = 5                  # Change an element of the array
print(a)                  # Prints "[5, 2, 3]"
​
b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 (2 dimensional) array
print(b.shape)                     # Prints "(2, 3)"
print(b[0, 0], b[0, 1], b[1, 0])   # Prints "1 2 4"
For the complete guide please refer to "http://cs231n.github.io/python-numpy-tutorial/#numpy"
iv. Pandas
Pandas bascially provides fast, flexible and expressive data structrures which is very helpful when working with relational or labled data. pandas is well suited for many different kinds of data: a. Tabular data with heterogeneously-typed columns, as in an SQL table or Excel spreadsheet b. Ordered and unordered (not necessarily fixed-frequency) time series data c. Arbitrary matrix data (homogeneously typed or heterogeneous) with row and column labels Any other form of observational / statistical data sets d. The data actually need not be labeled at all to be placed into a pandas data structure
The two primary data structures of pandas, Series (1-dimensional) and DataFrame (2-dimensional), handle the vast majority of typical use cases in finance, statistics, social science, and many areas of engineering

import pandas as pd
s = pd.Series([1,3,5,np.nan,6,8])  # generate pandas series with nan values
dates = pd.date_range('20130101', periods=6) # Generate a data frame of dates starting from 20130101 and the next 5 days
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD')) # Generate a dataframe with 4 columns of random number and make the dates as its index
df2 = pd.DataFrame({ 'A' : 1.,
'B' : pd.Timestamp('20130102'),
'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
'D' : np.array([3] * 4,dtype='int32'),
'E' : pd.Categorical(["test","train","test","train"]),
'F' : 'foo' })   # Generate a dataframe by passing a dict of objects that can  be converted to series like
For the complete guide refer to https://pandas.pydata.org/pandas-docs/stable/10min.html & http://pandas.pydata.org/pandas-docs/stable/
v. sci-kit learn
Scikit-learn provides a range of supervised and unsupervised learning algorithms via a consistent interface in Python. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy. I have also used sci-kit learn for data imputation, scaling, feature selection etc.

# (pip install sklearn)
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn import preprocessing
For reference this link can be used https://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/
vi. Matplotlib
Matplotlib is probably the single most used Python package for 2D-graphics. It provides both a very quick way to visualize data from Python and publication-quality figures in many formats. We are going to explore matplotlib in interactive mode covering most common cases.

from matplotlib import pyplot as plt
data[data.dtypes[(data.dtypes=="float")].index.values].hist(figsize=[20,20]) # for plotting histogram

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)
# Simple plot
import numpy as np
import matplotlib.pyplot as plt
​
X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)
​
plt.plot(X, C)
plt.plot(X, S)
​
plt.show()
<Figure size 640x480 with 1 Axes>
The link mentioned below is very helpful http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
