# Building a Movie Recommender System in Python
## What is a movie recommendation system ?
As the name suggests Recommendation System is a piece of software that recommends us some products or services.
Recommender system is used immensely in Today's world , from Amazon.com , Netflix to Google search ,so many other systems use this as a service.

![Image](https://user-images.githubusercontent.com/67604006/87637187-80814f00-c75f-11ea-93dd-6dc9dd94ded4.jpg)


## Types of Recommendation System
- **Popularity Based Recommender System**

This kind of system recommends the most popular items to the users. Most popular items is the item that is used or liked by most number of users.
For example, youtube trending list or Trending stuffs on myntra.
 
![Image](https://user-images.githubusercontent.com/67604006/87638039-bf63d480-c760-11ea-9fc3-28c3e3936951.png)


- **Content Based Recommender System**

Such recommendation system suggests similar kind of products used by the users . For example Netflix recommends us the similar movies to the movie we recently watched.
Similarly, Youtube also recommends us similar videos to the videos in our watch history.

The model works on this kind of principle:

![Image](https://miro.medium.com/max/500/1*x8gTiprhLs7zflmEn1UjAQ.png)

- **Collaborative Filtering based Recommender System**
Collaborative Filtering based recommender system creates profiles of users based on the items the user likes. 
Then it recommends the items liked by a user to the user with similar profile.For example, Google creates our profile based on our browsing history and then shows us the relevant ads.

![Image](https://user-images.githubusercontent.com/67604006/87639429-e3281a00-c762-11ea-802a-7dca6a50b260.png)

We will try to build Content Based Recommendation System.

# Content Based Recommender System
Content Based Recommender System recommends items similar to the items user likes. How do we decide which item is most similar to what user likes.
Here comes Similarity score.

- **Similarity Score**

It is a numerical value ranging from 0 to 1 . It determines how much two items are related to each other on scale of zero to one . This
similarity score is determined by comparing the text details of both the items.So, similarity score is the measure of similarity between given text details of two items.

We will be using cosine similarity between texts details of items. Follow these steps :
1. Count the number of unique words in both texts.
2. Count the frequency of each word in each text.
3. Find the points of both texts and get the value of cosine distance between them.
Example : Text1 = " ALL  WELL" Text2 ="WELL ALL WELL". There are 2 unique words here "All" and "Well"
4. Counting the frequency.

![1](https://user-images.githubusercontent.com/67604006/87642773-96930d80-c767-11ea-93ce-f90c4a405fa7.png)

5. There are only two unique words, hence we’re making a 2D graph. Now we need to find the cosine of angle ‘a’, which is the value of angle between both vectors. 

![Image](https://www.oreilly.com/library/view/statistics-for-machine/9781788295758/assets/2b4a7a82-ad4c-4b2a-b808-e423a334de6f.png)

Here is the formula used for doing this:

![Image](https://www.academyofdatascience.com/images/blog_images/py7%20(1).jpg)

The value of cos(a) define the similarity between the texts.

# Let's build our python model
We are going to make a Hollywood Recommendation System in Python. You can download the data set from this [Link](https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset) .
We will do some standard imports.

```ruby
import numpy as np
import pandas as pd
```
Read the csv file

```ruby
data = pd.read_csv('movie_metadata.csv')
```
Have look at your dataset

```ruby
data.head()
```
![Image](https://user-images.githubusercontent.com/67604006/87645222-cabbfd80-c76a-11ea-81e6-1eab5abf60e5.png)

Let us check null values.

```ruby
data.isnull().sum(axis=0)
```
![Image](https://user-images.githubusercontent.com/67604006/87653836-baf4e700-c773-11ea-9e0b-0758cc729663.png)

replacing null values in the all columns with string 'unknown'
```ruby
data['actor_1_name'] = data['actor_1_name'].replace(np.nan, 'unknown')
data['actor_2_name'] = data['actor_2_name'].replace(np.nan, 'unknown')
data['actor_3_name'] = data['actor_3_name'].replace(np.nan, 'unknown')
data['director_name'] = data['director_name'].replace(np.nan, 'unknown')
```
In the ‘genres’ column, replacing the ‘|’ with whitespace, so the genres would be considered different strings

```ruby
data['genres'] = data['genres'].replace('|', ' ')
```
Now converting the ‘movie_title’ columns values to lowercase for searching simplicity.
```ruby
data['movie_title'] = data['movie_title'].str.lower()
```
One more operation we need to perform on the title column. All the title strings has one special character in the end, which will cause problem while searching for the movie in dataset. Hence removing the last character from all the elements of ‘movie_title’.

```ruby
data['movie_title'][0]
```
```ruby
data['movie_title'] = data['movie_title'].str[:-1]
```
keeping the columns that are useful in recommndation system

```ruby
data = data.loc[:,['actor_1_name','actor_2_name','actor_3_name','director_name','genres','movie_title']]
```
saving this file
```ruby
data.to_csv('data.csv',index=False)
```
libraries for making count matrix and similarity matrix
```ruby
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```
making the new column containing combination of all the features

```ruby
data['comb'] = data['actor_1_name'] + ' ' + data['actor_2_name'] + ' '+ data['actor_3_name'] + ' '+ data['director_name'] +' ' + data['genres']
```
creating a count matrix
```ruby
cv = CountVectorizer()
count_matrix = cv.fit_transform(data['comb'])
```
creating a similarity score matrix
```ruby
sim = cosine_similarity(count_matrix)
```
saving the similarity score matrix in a file for later use
```ruby
np.save('similarity_matrix', sim)
```
saving dataframe to csv for later use in main file
```ruby
data.to_csv('data.csv',index=False)
```
 libraries for making count matrix and similarity matrix
 ```ruby
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```
define a function that creates similarity matrix,if it doesn't exist
```ruby
def create_sim():
    data = pd.read_csv('data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    sim = cosine_similarity(count_matrix)
    return data,sim
```
defining a function that recommends 10 most similar movies
```ruby
def rcmd(m):
    m = m.lower()
    # check if data and sim are already assigned
    try:
        data.head()
        sim.shape
    except:
        data, sim = create_sim()
    # check if the movie is in our database or not
    if m not in data['movie_title'].unique():
        return('This movie is not in our database.\nPlease check if you spelled it correct.')
    else:
        # getting the index of the movie in the dataframe
        i = data.loc[data['movie_title']==m].index[0]

        # fetching the row containing similarity scores of the movie
        # from similarity matrix and enumerate it
        lst = list(enumerate(sim[i]))

        # sorting this list in decreasing order based on the similarity score
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)

        # taking top 1- movie scores
        # not taking the first index since it is the same movie
        lst = lst[1:11]

        # making an empty list that will containg all 10 movie recommendations
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l
```

```ruby
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = rcmd(movie)
    movie = movie.upper()
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l')



if __name__ == '__main__':
    app.run()
```
Click the [this](https://github.com/ShivamKumar-bit/MVRS) link to view the Github code.

