
## Understanding Graphical Social Networking concepts for use in ML

Social Network these days are a part of our everyday life which paves the way for using social network analysis as a tool to solving wide variety of issues in the industry. In a recent [article](https://www.nytimes.com/2019/06/02/us/us-visa-application-social-media.html) published by the NY Times, the US immigrant authorities are using the social media information of its applicants in order to grant them a visa status in the country. Millions of applications are processed every year, making it next to impossible to sit through every user's social media profiles and analyse their credibility!

Now, Imagine yourself working for the US authorities. You are approached by your client to find a possible solution to this problem. Your task is to extract information(features) from a sample of dataset to find the most potential or eligible candidates. 

As a first part of this assignment, we will try to find a way to solve this using some critical measures in graphical network analysis. Our dataset [facebook_edges.csv] contains a graph with nodes as your applicants, and edges representing their connections. 

After completing this assignment, you should be able to answer the following questions:

### Graphical Network: Centrality Measures
* What are the measures of centrality? 
* What are the different types of most commonly used centrality measures?
* Why are they important for a graphical networks ? 

#### Notes: 
* Information on the dataset : The data collected from Facebook is in the following file : 
      1. facebook_edges.csv : contains networking information of these profiles. Each record is in format "a b" denoting 'a' is facebook friend of 'b' and vice-versa. Also, the graph can be assumed to be 'UNDIRECTED'.
       
* For ease of setup and simplicity, I would recommend using [networkx](https://networkx.github.io/). However, there are few other tools like [graph-tool](https://graph-tool.skewed.de/static/doc/index.html), [gephi](https://gephi.org/), [netwulf](https://github.com/benmaier/netwulf/) etc. Feel free to use any tool of your choice. 

* You can either use Spark DataFrame or pandas.DataFrame to do the assignment. In comparison, pandas.DataFrame has richer APIs, but is not good at distributed computing.

## Part 1: Centrality Measures

In graph theory and network analysis, indicators of centrality identify the most important vertices within a graph. Applications include identifying the most influential person(s) in a social network. These centrality measures in graphical network answers a very basic question : "Who is the most important or central person in the network?". Before proceeding ahead, I would strongly recommend you to read centrality and its different measures [online](https://en.wikipedia.org/wiki/Centrality).

In our scenario, we will try using centrality as a tool for identifying most prominent applicants. 

For the first step, we read the data as DataFrames and further implement algorithms related to centrality measures to understand their importance in graphical analysis. 


```python
import os
import pandas as pd
#<-- Write Your Code -->
source = "dataset/"

#Read the edges file as dataframe
df_edges = pd.read_csv(source+"facebook_edges.csv")
df_edges.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>node0</th>
      <th>node1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>881</td>
      <td>858</td>
    </tr>
    <tr>
      <th>1</th>
      <td>828</td>
      <td>697</td>
    </tr>
    <tr>
      <th>2</th>
      <td>884</td>
      <td>864</td>
    </tr>
    <tr>
      <th>3</th>
      <td>856</td>
      <td>869</td>
    </tr>
    <tr>
      <th>4</th>
      <td>889</td>
      <td>856</td>
    </tr>
    <tr>
      <th>5</th>
      <td>872</td>
      <td>873</td>
    </tr>
    <tr>
      <th>6</th>
      <td>719</td>
      <td>713</td>
    </tr>
    <tr>
      <th>7</th>
      <td>861</td>
      <td>863</td>
    </tr>
    <tr>
      <th>8</th>
      <td>840</td>
      <td>803</td>
    </tr>
    <tr>
      <th>9</th>
      <td>864</td>
      <td>856</td>
    </tr>
  </tbody>
</table>
</div>



Now, create a labelled graph from about `1000` random samples of the dataframe and save the image as A.png. (For higher number of nodes this steps takes time.) 


```python
from networkx import *
import matplotlib.pyplot as plt

#<-- Write Your Code -->
# Create graph for 1000 random samples. 



plt.savefig("A.png", format="PNG",figsize=(10,10))
```

### Task 1a: Degree Centrality
For this step we will find `Degree centrality measure`, which is one of the simplest centrality measure to compute. A node's degree is simply a count of how many social connections (i.e., edges) it has. To compute degree centrality you need to divide this value by `N-1`. Write a code to find the degree centrality of each node.  
* **Please note** you are **not allowed** to use **in-built function** for computing degree_centrality in this step. Try to implement it yourself.  
* Store the values for each node, in a new dataframe with a column `degree_centrality`, index being the vertices.


```python
#<-- Write Your Code -->

def get_degree_centrality(g,node,n):
    ''' Function returns degree centrality measure for each vertex (node)
    Args:
    g (obj): graph object  
    node (int): vertex index 
    n (int)  : total number of nodes in the graph minus 1
    Returns:
    Float value denoting the degree measure of the vertex
    '''
    return 


nodes = g.nodes()
df = pd.DataFrame(index=nodes)
N= len(nodes)
df['degree_centrality'] = df.index.map(lambda x: get_degree_centrality(g,x,N-1))
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>degree_centrality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2446</th>
      <td>0.003019</td>
    </tr>
    <tr>
      <th>2055</th>
      <td>0.003774</td>
    </tr>
    <tr>
      <th>1912</th>
      <td>0.004528</td>
    </tr>
    <tr>
      <th>2038</th>
      <td>0.001509</td>
    </tr>
    <tr>
      <th>1668</th>
      <td>0.001509</td>
    </tr>
  </tbody>
</table>
</div>



**Display the top 10 nodes with highest degree centrality values in a graph. Save this image as B.png.**


```python
#<-- Write Your Code -->
## Get the top nodes with high centrality 
max_nodes = 10
high_degree_nodes = (df['degree_centrality'].sort_values(ascending=False)[:max_nodes]).index

#<-- Write Your Code -->
## Create graph from these 10 nodes

#plt.savefig("B.png", format="PNG")
```

Degree centrality represents a measure of how many connections a person has. Higher values mean the user has more connections and therefore, is highly "important" for our analysis. However, imagine a spam user who happens to have many fake friends, has a high degree centrality value. In such scenarios, these numbers could really be misleading. A user's importance should not only depend upon how many connections it has but also how many connections its neighbors have (and how many connections its neighbors' neighbors have and so on). Identifying whether he is a `key` connector in the graph is also important. Going further, we use two more critical measures : Betweenness and Eigenvector centrality. We will notice that each centrality measure indicates a different type of node importance.  

### Task 1b : Betweenness (node) centrality

One another metric is betweenness centrality, which identifies people who frequently are on the shortest paths between pairs of other people. Recall [Assignment 6 from CMPT 732](https://coursys.sfu.ca/2018fa-cmpt-732-g1/pages/Assign6) where we implemented a parallelized version of Dijkstra's Algorithm in Spark. As a part of this segment, we will implement a simple algorithm to find the node centrality, i.e the number of shortest paths that each node is a part of. Centrality of node `i` is computed by adding up, for every other pair of nodes `j` and `k`, the number of shortest paths between node `j` and node `k` that pass through `i`. Please complete the function below that takes in the graph and returns a dictionary with their node centrality values. Note that **you are allowed** to use in-built functions for getting the shortest path between two nodes. 


```python
#<-- Write Your Code -->

def get_node_centrality(graph):
    """ Funtion returns a dictionary with each vertex as index and 
    no. of shortest paths that the vertex is a part of as its value.
    Args: 
    graph : graph object
    Return:
    n_spaths : (dict) vertex as keys and values as no. of shortest paths
    """
    # 1. initialize dictionary with 0.0
    n_spaths = dict.fromkeys(graph,0.0)
    
    # 2. get the shortest paths from all nodes. 
    s= dict(all_pairs_shortest_path(graph))
    
    # 3. write logic for counting the number of shortest paths each node is a part of. 
    
    return n_spaths

## Adding this value to original dataframe. 
node_centrality = get_node_centrality(g)

from sklearn.preprocessing import minmax_scale
df['node_centrality'] = df.index.map(lambda x: node_centrality[x])
df['node_centrality'] = minmax_scale(df['node_centrality'])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>degree_centrality</th>
      <th>node_centrality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2446</th>
      <td>0.003019</td>
      <td>0.453952</td>
    </tr>
    <tr>
      <th>2055</th>
      <td>0.003774</td>
      <td>0.604572</td>
    </tr>
    <tr>
      <th>1912</th>
      <td>0.004528</td>
      <td>0.441150</td>
    </tr>
    <tr>
      <th>2038</th>
      <td>0.001509</td>
      <td>0.059961</td>
    </tr>
    <tr>
      <th>1668</th>
      <td>0.001509</td>
      <td>0.071979</td>
    </tr>
  </tbody>
</table>
</div>



Betweenness centrality is a **normalized form** of nodal centrality measure. It is computed by adding up, for every other pair of nodes `j` and `k`, the `proportion` of shortest paths between node `j` and node `k` that pass through `i`. I would highly reccommend you to read about this measure [ online ](https://en.wikipedia.org/wiki/Betweenness_centrality). 


```python
### Finding betweenness centrality
bet = betweenness_centrality(g, normalized=True)
df['betweenness_centrality'] = df.index.map(lambda x: bet[x])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>degree_centrality</th>
      <th>node_centrality</th>
      <th>betweenness_centrality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2446</th>
      <td>0.003019</td>
      <td>0.453952</td>
      <td>0.001703</td>
    </tr>
    <tr>
      <th>2055</th>
      <td>0.003774</td>
      <td>0.604572</td>
      <td>0.002668</td>
    </tr>
    <tr>
      <th>1912</th>
      <td>0.004528</td>
      <td>0.441150</td>
      <td>0.001838</td>
    </tr>
    <tr>
      <th>2038</th>
      <td>0.001509</td>
      <td>0.059961</td>
      <td>0.000174</td>
    </tr>
    <tr>
      <th>1668</th>
      <td>0.001509</td>
      <td>0.071979</td>
      <td>0.000251</td>
    </tr>
  </tbody>
</table>
</div>



### Task 1c: EigenVector Centrality

As we saw before, computing shortest paths is kind of a pain. For this reason, node and betweenness centrality isn't often used on large networks. The less intuitive (but generally easier to compute) eigenvector centrality is more frequently used. Eigenvector centralities are numbers, one per user, such that each user’s value is a constant multiple of the sum of his neighbors’ values. In this case centrality means being connected to people who themselves are central. The more centrality you are directly connected to, the more central you are. This is of course a circular definition — eigenvectors are the way of breaking out of the circularity. I highly recommend you to read further about eigenvectors and their role in finding the centrality measure. **Please note**, for this task you can use the in-built function for computing eigenvector centrality but before that, please make sure you understand how it works. 


```python
#<-- Write Your Code -->
# Find EigenVector centrality (Remember to choose a high iteration value for `max_iter` for the algorithm to converge)

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>degree_centrality</th>
      <th>node_centrality</th>
      <th>betweenness_centrality</th>
      <th>eighenvector_centrality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.001509</td>
      <td>0.002743</td>
      <td>0.000009</td>
      <td>2.498123e-18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000755</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.102622e-41</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.000755</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.102622e-41</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.001509</td>
      <td>0.001960</td>
      <td>0.000006</td>
      <td>2.196813e-18</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0.000755</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.102622e-41</td>
    </tr>
  </tbody>
</table>
</div>



##### To inspect the difference between various centrality values we plot the top 100 users with highest degree centrality showing how these numbers can be different for different users. To give an idea, your graph should be somewhat like the plot given below.  

**Please Note** : Generally, the centrality numbers aren’t that meaningful themselves. What we really care about is how the numbers for each node compare to the numbers for other nodes. Hence, for the purpose of plot you are allowed to multiply centrality values by a common factor to bring it in a reasonable range. 


```python
#<-- Write Your Code -->


```


![png](output_23_0.png)


**Observe the graph closely and write two most interesting insights from the above comparison plot.**
* [1]  

* [2] 

In this part of the assignment, we focussed on finding the critical users (nodes) by establishing a connection between their centrality measures in graphical networks. Apart from US immigration, you can think of many other scenarios where this method can be useful. For example, say you are in the recruiting team of your company which is looking to hire data scientists. Since there are a lot of applicants, instead of just going through their resumes (since they all look the same anyway! ;) ), you plan on using their coorporate social media profiles (on linkedin) for finding the most "renowned" data scientists for your company. 

Likewise, you can think about more such scenarios which can be solved using the above approach for graphical networks.  
