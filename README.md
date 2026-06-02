# Instructions

1. Install pytorch
https://pytorch.org/get-started/locally/

2. Install pytorch geometric
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

3. Install `torch-cluster`
https://github.com/rusty1s/pytorch_cluster

NOTE: Using CPU for all methods is completely fine.

4. Install joern, our library for CPG generation. The script assumes binaries are in path.
https://docs.joern.io/installation/

I've included case-07 of https://github.com/oscarkarnalim/sourcecodeplagiarismdataset/tree/master as code samples

collusion_space_mapping.png is the example output from the code.

The program takes in two paths as arguements, these are examples:

--dataset_dir IR-Plag-Dataset/ --out_dir=output/


5. Final command example

python --dataset_dir IR-Plag-Dataset/ --out_dir=output/
