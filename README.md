# BilinearReview_journal
Personalized Sentiment Classification of Customer Reviews via an Interactive Attributes Attention Model

## Main Requirements
```
transformers
yaml
easydict
torchtext
spacy
```


## Simple Usage
- all datasets should be downloaded and unzipped in the directory `corpus` in the orginal codes

    - Original IMDB, Yelp datasets are available at [here](http://ir.hit.edu.cn/~dytang/paper/acl2015/dataset.7z) and Amazon dataset at [here](https://nijianmo.github.io/amazon/index.html)
    - zipped datasets are also downloaded at [here](https://pan.baidu.com/s/1KrMU6aOd8xGstw9CCBP0Yg) (提取码：luck)

- running a model over IMDB datasets
```
python run.py --run train --dataset imdb --model bilinear --gpu 0,1
```

## Noting
All training Hyper-Parameters are set in cfgs/config.py and model parameters in cfgs/bilinear_model.yml.
