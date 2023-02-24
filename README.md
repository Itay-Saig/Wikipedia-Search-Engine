# Information Retrieval project - Wikipedia search engine
Building a search engine on the corpus of the entire English Wikipedia (more than 6 million documents).
The final engine ran on a GCP virtual machine.

This repository includes the following files:
1. Create_Indices.ipynb - a notebook ran on GCP to build the indices (title index, body index and anchor index).
2. Inverted_index_gcp.py - a python implementation of an Inverted Index class.
3. train_queries.json - a training set, a json file with queries and top (most relevant) wiki pages excpected.
4. run_frontend_gcp.ipynb - a notebook to run tests on the whole corpus on a gcp cluster.
5. run_frontend_in_colab.ipynb - a notebook to run tests on a small corpus (1000 documents only).
6. search_frontend.py - the backend python script for querying the search engine, containing 5 methods of querying: wiki doc body, wiki title, wiki anchor text, PageRank, PageViews, and more helper functions which merged into a general search query function.
7. updated_inverted_index.py - an updated class of Inverted Index with some more functionalitis.
8. run_queries.py - a notebook that contains a code that implements Map@40 score and an iterative process to run the queries in the training set, calculate scores and measure run times.
9. Project Report.pdf - final report about the project and the findings.
10. Project Presentation - a short presentation summarizing the project and the findings.
