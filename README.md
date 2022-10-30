README
============================
This is the source code of our paper.

## Dataset:
* The links of the original datasets ICEWS14 and ICEWS05-15 can be found from paper: [Learning Sequence Encoder for Temporal Knowledge Graph Completion](https://github.com/nle-ml/mmkb).
* YAGO11k and Wikidata12k have been preprocessed with reference to [Temporal Knowledge Graph Completion Based on Time Series Gaussian Embedding](https://github.com/soledad921/ATISE).

## Usage:
* model.py contains PTorch(1.x) based implementation of our proposed models.
* To reproduce the reported results of our model, use the following commands:

      python main.py --model SANe --name sane --lr 0.001 --data icews14 --train_strategy one_to_x
    
      python main.py --model SANe --name sane --lr 0.001 --data icews05-15 --k_h 30 --embed_dim 300 --feat_drop 0.2 --hid_drop 0.3 --ker_sz 7 --train_strategy one_to_x --batch 512

      python main.py --model SANe --name sane --lr 0.001 --data yago --ker_sz 5 --train_strategy one_to_x

      python main.py --model SANe --name sane --lr 0.001 --data wikidata --ker_sz 5 --train_strategy one_to_x

    
