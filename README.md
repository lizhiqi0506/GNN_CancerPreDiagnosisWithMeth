## Graph neural network learning model for precision multi-tumour early diagnostics with DNA methylation data

---

### Running Environment

* Linux environment, Python 3
* The following packages need to be available: numpy,pandas,collections,pytorch,torch_geometirc,seaborn
* The dataset created by torch_geometric must be saved in a dirctory named "processed"

### Install
* Some basic packages can be installed via pip directly

  ```
  pip install numpy
  pip install pandas
  pip install collections
  pip install seaborn
  ```

* When installing pytorch, the versions of pytorch, torchvision, cuda and cudnn should be matched. Here is my install command as a reference.
  ```
  conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
  ```
* torch_geometric is a powerful API of many kinds of Graph Neural Network based on Pytorch. There are 4 dependency packages to be installed before installing torch_geometric, and those version should be matched with the versions of cuda and pytorch.
  
  ```
  $ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
  $ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
  $ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
  $ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
  $ python setup.py install or pip install torch-geometric
  ```

### Prediction

In this project, 3 kinds of classification are performed with GNN:

1. Create graphs from samples of different kinds of tissues respectively where a node stands for a methylation sites, the feature of the node is the beta values of the site in different samples, the existence of edge depend on the correlation coefficient between two nodes' features and the feature of edge is relative to the correlation coefficient, too. Then we use GNN to classify which tissue one graph belongs to and more importantly, extract the most activated methylation sites that perform differently in different tissues for further research.
2. From the samples of same tissue, sampling tumor samples and normal samples to create graphs respectively. The method of creating a graph is similar to that in (1). Using GNN to classify whether one graph belongs to tumor or not and extract the methylation sites that perform differently between tumor and normal samples.
3. Using GNN to classify which tissue one sample belongs to: create graphs where a node stands for a sample, the feature of the node is the beta values of the sites selected from (1) of the sample the existence of edge depend on the correlation coefficient between two nodes' features and the feature of edge is relative to the correlation coefficient, too. Then using GNN to perform node-classification.

The prediction.py can be run as follows:
```
python prediction.py --mode/-m [tissue, case-normal, node] --input/-i <input_file_name> --output/-o <output_file_name> --tissue/-t <tissue>
```
"--mode/-m" argument is the classification mode, tissue, case-normal and node stand for 1,2 and 3 mentioned above respectively. "--tissue/-t" is used when "--mode/-m" argument is set as "case-normal". When "--mode/-m" argument is set as "node", the input file should be an Illumina Methylation 450k .txt file, otherwise the input file should be a .pt file created by createDataset-script.

To get more details, you can type:
```
python prediction.py --help
```

For example:

```
python prediction.py -m node -i test/test.txt -o test/test.out
```



    

  
