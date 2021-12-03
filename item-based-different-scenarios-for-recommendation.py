#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # MOVIE RECOMMENDATION SYSTEMS 
# ## 1. BASED ON USER VOTES IN THE LAST 10 YEARS

# In[2]:


import pandas as pd
pd.set_option('display.max_columns', 30)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# pd.set_option('display.width', 200)


# In[3]:


import pandas as pd
pd.set_option('display.max_columns', 20)

movie = pd.read_csv('../input/moviee-rating/movie.csv')
rating = pd.read_csv('../input/moviee-rating/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df = df.iloc[:10000000, :]


# In[4]:


df.head()


# # Preparing Data
# 

# In[5]:


df['year_movie'] = df.title.str.extract('(\(\d\d\d\d\))', expand=False)
df['year_movie'] = df.year_movie.str.extract('(\d\d\d\d)', expand=False)
df['title'] = df.title.str.replace('(\(\d\d\d\d\))', '')
df['title'] = df['title'].apply(lambda x: x.strip())

#################
# genres
#################

df["genre"] = df["genres"].apply(lambda x: x.split("|")[0])
df.drop("genres", inplace=True, axis=1)

#################
# timestamp
#################

df["timestamp"] = pd.to_datetime(df["timestamp"], format='%Y-%m-%d')
df.info()

df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day


# In[6]:


df.head()


# In[7]:


df.shape


# # Creating User_movie Matrix (user_movie_df)

# In[8]:


# rare movies

rare_movies = df['title'].value_counts() < 1000
rare_movies = rare_movies.index
print(rare_movies)


# In[9]:


a = pd.DataFrame(df["title"].value_counts())
rare_movies = a[a["title"] <= 1000].index


# In[10]:


# Before creating matrix rare movies have been removed from data set

common_movies = df[~df["title"].isin(rare_movies)]
common_movies = common_movies.iloc[:1000000, :]
print(common_movies)


# In[11]:


# According to people who voted last 10 years

a = df['year'].max() - 10
df[df['year'] > a].head()


# In[12]:


# Creating matrix with pivot table

user_movie_df = common_movies.pivot_table(index = ['userId'], columns = ['title'], values = 'rating')


# In[13]:


user_movie_df.head()


# In[14]:


movie = user_movie_df['Babe']


# In[15]:


user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)


# # THE BASIC ASSUMPTION HERE IS THAT "RECOMMENDATIONS SHOULD CHANGE AFTER THE USER ENTERS THE CATEGORY.
# ## 2. Analysis is made only for the category of movie 'Babe' , would recommendation change ?

# In[16]:


genre_df = df[df['genre'] == 'Children']
print(genre_df)


# In[17]:


user_movie_df = genre_df.pivot_table(index=["userId"], columns=["title"], values="rating")


# In[18]:


movie = user_movie_df['Babe']


# In[19]:


user_movie_df.corrwith(movie).sort_values(ascending = False).head()


# ## 3. Examining movies similar to our movie through categories other than the Action category

# In[20]:


genre_df = genre_df.copy()
genre_df.loc[genre_df["title"] == 'Babe', "genre"] = "new_children"
print(genre_df)


# In[21]:


genre_df[genre_df['title'] =='Babe'].head()


# In[22]:


user_movie_df = genre_df.pivot_table(index=["userId"], columns=["title"], values="rating")


# In[23]:


movie = user_movie_df['Babe']


# In[24]:


user_movie_df.corrwith(movie).sort_values(ascending = False).head(10)

