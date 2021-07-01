# BSD-GCN
The tensorflow code of the paper"Dual-Graph Convolutional Network Based on Band Attention and Sparse Constraint for Hyperspectral Band Selection"
Users can directly run main.py to train the BSD-GCN on Indian Pines.
If users hope to handle with new datasets, please switch to pre_data\sample_cuda, input "python setup.py install" in terminal and uncomment the "Get_graph_file(data_set)" in line 10 of utils. It is used to build the graph on HSI dataset.
The new dataset and labels should be put in data/ori_data.
