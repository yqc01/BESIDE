# ``Bridge” Enhanced Signed Directed Network Embedding

**Introduction**

This repository provides a reference implementation of BESIDE as described in the paper: <br>
> ``Bridge” Enhanced Signed Directed Network Embedding.<br>
> Yiqi Chen, Tieyun Qian, Huan Liu and Ke Sun.<br>
> Conference on Information and Knowledge Management, 2018.<br>


### Basic Usage

#### Example
1.preprocess public dataset(download from [here](http://snap.stanford.edu/data/index.html)):
```
python preprocess_data.py slashdot ./original_dataset/soc-sign-Slashdot090221.txt ./dataset/soc-sign-Slashdot090221.txt.map ./dataset/soc-sign-Slashdot090221.txt.map.train ./dataset/soc-sign-Slashdot090221.txt.map.test ./dataset/soc-sign-Slashdot090221.txt.map.nodes
```

2.train + test model:<br/>

```
python BESIDE_train.py slashdot tri_sta 20 100 ./dataset/soc-sign-Slashdot090221.txt.map.train ./dataset/soc-sign-Slashdot090221.txt.map.test ./dataset/soc-sign-Slashdot090221.txt.map.nodes
```

test results are written in log/\<some-name>.log

#### Options
1.preprocess

```
Usage:
python preprocess_data.py <dataset_choose> <dataset_fpath_origin> <dataset_fpath_output> <dataset_train_fpath> <dataset_test_fpath> <dataset_nodes_fpath>

arguments:
dataset_choose: select a name for your dataset (e.g. slashdot, epinions, wikirfa)
dataset_fpath_origin: input dataset file path
dataset_fpath_output: output dataset file path
dataset_train_fpath: output train file path
dataset_test_fpath: output test file path
dataset_nodes_fpath: output nodes id file path
```

2.train + test (link sign prediction)

```
Usage: 
python BESIDE_train.py <dataset_choose> <mode_choose> <emb_dim> <epoch_num> <dataset_train_fpath> <dataset_test_fpath> <dataset_nodes_fpath>

arguments:
dataset_choose: select a name for your dataset (e.g. slashdot, epinions, wikirfa)
mode_choose: three mode, [tri_sta, tri, sta]
emb_dim: embedding dimension for nodes
epoch_num: train epoch number
dataset_train_fpath: train file path (you can use preprocess_data to generate it), edgelist format
dataset_test_fpath: test file path (you can use preprocess_data to generate it), edgelist format
dataset_nodes_fpath: nodes id file path (you can use preprocess_data to generate it)
```

3.test(status comparison)

```
Usage: 
python status_comp.py <method_choose> <dataset_choose> <dataset_train_fpath> <dataset_test_fpath> <dataset_nodes_fpath> (<BESIDE_para_fpath>)

arguments:
method choose: select model method name ([BESIDE_tri_sta, BESIDE_sta, pagerank])
dataset_choose: select a name for your dataset (e.g. slashdot, epinions, wikirfa)
dataset_train_fpath: train file path (you can use preprocess_data to generate it), edgelist format
dataset_test_fpath: test file path (you can use preprocess_data to generate it), edgelist format
dataset_nodes_fpath: nodes id file path (you can use preprocess_data to generate it)
BESIDE_para_fpath: BESIDE model parameters file path

example:
python status_comp.py BESIDE_tri_sta slashdot ./dataset/soc-sign-Slashdot090221.txt.map.train ./dataset/soc-sign-Slashdot090221.txt.map.test ./dataset/soc-sign-Slashdot090221.txt.map.nodes ./emb/example.emb.ep10
```

#### Input

edgelist format of signed directed network:
	<node1_id> <node2_id> \<sign>

detail can be drawn from [here](http://snap.stanford.edu/data/soc-sign-Slashdot090221.html).

#### Output
test results are in log/ directory;

model parameters are saved in pickle format, you can load it following read_emb_para_example.py.

### Requirement
```
python3 (3.5.3)
tensorflow >= 1.1.0
networkx >= 2.0
sklearn >= 0.19.1
numpy >= 1.14.5
```

