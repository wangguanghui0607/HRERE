HRERE
=======

Connecting Language and Knowledge with Heterogeneous Representations for Neural Relation Extraction

Paper Published in NAACL 2019: [HRERE](https://arxiv.org/abs/1903.10126)

### Prerequisites

- tensorflow >= r1.2
- hyperopt
- gensim
- sklearn

### Dataset

To download the dataset(NYT corpus, Fb3m, Glove embeddings) used:

```
cd ./data
python prepare_data.py
```
#### NYT corpus
[The New York Times Annotated Corpus](https://catalog.ldc.upenn.edu/LDC2008T19), the sample data as below.
```
m.01l443l	m.04t_bj	dave_holland	barry_altschul	NA	the occasion was suitably exceptional : a reunion of the 1970s-era sam rivers trio , with dave_holland on bass and barry_altschul on drums .	###END###
m.01l443l	m.04t_bj	dave_holland	barry_altschul	NA	tonight he brings his energies and expertise to the miller theater for the festival 's thrilling finale : a reunion of the 1970s sam rivers trio , with dave_holland on bass and barry_altschul on drums .	###END###
```

#### Fb3m
dataset from [Freebase](https://developers.google.com/freebase/) dump released of June 2015, the sample data is as below.
```
m.03frx9g	/music/release/region	m.09c7w0
m.0q8hxch	/music/release/format	m.01www
```
#### Glove
[GloVe](https://nlp.stanford.edu/projects/glove/) is an unsupervised learning algorithm for obtaining vector representations for words.

### Preprocessing

#### Construct the knowledge graph.
Create dataset same format(MID, relation, MID) as fb3m for training:

```
python create_kg.py
```

#### Preprocessing the data:
Create dataset which fileds is [relation_id, entity_1_id, entity_1_pos_start, entity_1_pos_end, entity_2_id, entity_2_pos_start, entity_2_pos_end, sentence]

```
python preprocess.py -p -g
```

### Complex Embeddings

Copy the directory `./fb3m` in the `data` folder in [tensorflow-efe](https://github.com/billy-inn/tensorflow-efe) and run the following commands to obtain the complex embeddings:

```
# Convert raw data into dataset replaced by entity id and relation id
python preprocess.py --data fb3m
python train.py --model best_Complex_tanh_fb3m --data fb3m --save
python get_embeddings.py --embed complex --model best_Complex_tanh_fb3m --output <repo_path>/fb3m
```

Then copy `e2id.txt` and `r2id.txt` in the `tensorflow-efe/data/fb3m` to `./fb3m` and run the following command:

```
python get_embeddings.py 
```

### Hyperparameters Tuning

```
python task.py --model <model_name> --eval <max_number_of_search> --runs <number_of_runs_per_setting>
```

`model_name` can be found in `model_param_space.py`. You can also define the search space by yourself.

### Evaluation

```
python eval.py --model <model_name> --prefix <file_prefix> --runs <number_of_runs>
```

`model_name` can be found in `model_param_space.py`. To replicate our results, use `best_complex_hrere` as the `model_name`.
It will run the model multiple times and calculate the means and stds of P@N which are logged in `./log`.
The predicted probabilities and labels of the first run are stored in `plot/output` for plotting PR curves.

### Results

![Curve](plot/figure/comparison.png)

After replicating the results, we find that the results on P@N(%) reported in the paper seem to be a bit over-optimisitic due to the variance.
According our replication based on 5 runs (`./log/replication.log`), the results are P@10% (0.849 +- 0.019), P@30% (0.728 +- 0.019), P@50% (0.636 +- 0.013).
We also report our scores to [NLP Progress](http://nlpprogress.com/english/relationship_extraction.html) based on this replication.

### Cite

If you found this codebase or our work useful, please cite:

```
@InProceddings{xu2019connecting,
  author = {Xu, Peng and Barbosa, Denilson},
  title = {Connecting Language and Knowledge with Heterogeneous Representations for Neural Relation Extraction}
  booktitle = {The 17th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL 2019)},
  month = {June},
  year = {2019},
  publisher = {ACL}
}
```
