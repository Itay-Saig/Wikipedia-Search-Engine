# IR-Project
Building a Wikipedia Search Engine.
In this project I built a search engine on more than 6M wikipedia docs.
The final engine ran on a GCP virtual machine.

This repository includes the following files:
1. Create_Indices.ipynb - A notebook we ran on GCP to build our indices.
2. Inverted_index_gcp.py - A python implementation of an Inverted Index class.
3. new_train.json - a training set, a json file with queries and top (most relevant) wiki pages excpected.
4. run_frontend_gcp.ipynb - a notebook to run tests on the whole corpus on a gcp cluster.
5. run_frontend_in_colab.ipynb - a notebook to run tests on a small corpus (1000 docs)
6. search_frontend.py - the backend python script for querying our search engine, containing 5 methods of querying: wiki doc body, wiki title, wiki anchor text, pagerank, # of page views and more helper functions which we united to a general search wuery function.
7. updated_inverted_index.py - an updated class of Inverted Index with some more functionalitis.
8. run_queries.py - a notebook that contains a code that implements Map@40 score and an iterative process to run the queries in the training set, calculate scores and measure run times.
9. IR Final Project Report.pdf - final report about the project.
