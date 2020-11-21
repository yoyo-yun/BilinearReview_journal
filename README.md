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
- all datasets have been attached in the directory `corpus` in the orginal codes
- running a model over IMDB datasets
```
python run.py --run train --dataset imdb --model bilinear --gpu 0,1
```

## Noting
All training Hyper-Parameters are set in cfgs/config.py and model parameters in cfgs/bilinear_model.yml.
