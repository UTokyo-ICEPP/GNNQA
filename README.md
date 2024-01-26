# GNNQA
Developing a GNN to calssify doublets in the QA tracking algorithm.

# modules/TrackDataloader.py
Dataloader, generate the graph in eta-phi space.

# modules/dynamic_graph_PhaseIII.py
Main code to build up the model. First transform the graph from Dataloader and generate a k-nn graph in the embedding space. At the end tranform it back to get the edge scores.

#training_PhaseIII.py
Main training code, in each loop we need to the the HitiD matching in order to trace the edge scores.

#model_test.py
Use for checking the strutuce of the training_PhaseIII.py, only contain the core of training without looping over the epochs.

#test_only_PhaseIII.py
For test run only, still developing from the old package.
