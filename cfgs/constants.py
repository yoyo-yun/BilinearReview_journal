import os
from datasets.imdb import IMDBHierarchical, IMDB
from datasets.yelp_13 import YELP13Hierarchical, YELP13
from datasets.yelp_14 import YELP14Hierarchical, YELP14
from datasets.digital import DIGITALHierarchical, DIGITAL
from datasets.industrial import Industrial, IndustrialHierarchical
from datasets.software import Software, SoftwareHierarchical
from models_v2.baselines import LSTM

PRE_TRAINED_VECTOR_PATH = 'pretrained_Vectors'
if not os.path.exists(PRE_TRAINED_VECTOR_PATH):
    os.makedirs(PRE_TRAINED_VECTOR_PATH)

DATASET_PATH = 'corpus'

DATASET_PATH_MAP = {
    "imdb": os.path.join(DATASET_PATH, 'imdb'),
    "yelp_13": os.path.join(DATASET_PATH, 'yelp_13'),
    "yelp_14": os.path.join(DATASET_PATH, 'yelp_14'),
    "digital": os.path.join(DATASET_PATH, 'amazon'),
    "industrial": os.path.join(DATASET_PATH, 'amazon'),
    "software": os.path.join(DATASET_PATH, 'amazon'),
}

TEXT_VECTORS_MAP = {
    "imdb": os.path.join(PRE_TRAINED_VECTOR_PATH, 'imdb_text.pt'),
    "yelp_13": os.path.join(PRE_TRAINED_VECTOR_PATH, 'yelp_13_text.pt'),
    "yelp_14": os.path.join(PRE_TRAINED_VECTOR_PATH, 'yelp_14_text.pt'),
    "digital": os.path.join(PRE_TRAINED_VECTOR_PATH, 'digital_text.pt'),
    "industrial": os.path.join(PRE_TRAINED_VECTOR_PATH, 'industrial_text.pt'),
    "software": os.path.join(PRE_TRAINED_VECTOR_PATH, 'software_text.pt'),
}
USR_VECTORS_MAP = {
    "imdb": os.path.join(PRE_TRAINED_VECTOR_PATH, 'imdb_usr.pt'),
    "yelp_13": os.path.join(PRE_TRAINED_VECTOR_PATH, 'yelp_13_usr.pt'),
    "yelp_14": os.path.join(PRE_TRAINED_VECTOR_PATH, 'yelp_14_usr.pt'),
    "digital": os.path.join(PRE_TRAINED_VECTOR_PATH, 'digital_usr.pt'),
    "industrial": os.path.join(PRE_TRAINED_VECTOR_PATH, 'industrial_usr.pt'),
    "software": os.path.join(PRE_TRAINED_VECTOR_PATH, 'software_usr.pt'),
}
PRD_VECTORS_MAP = {
    "imdb": os.path.join(PRE_TRAINED_VECTOR_PATH, 'imdb_prd.pt'),
    "yelp_13": os.path.join(PRE_TRAINED_VECTOR_PATH, 'yelp_13_prd.pt'),
    "yelp_14": os.path.join(PRE_TRAINED_VECTOR_PATH, 'yelp_14_prd.pt'),
    "digital": os.path.join(PRE_TRAINED_VECTOR_PATH, 'digital_prd.pt'),
    "industrial": os.path.join(PRE_TRAINED_VECTOR_PATH, 'industrial_prd.pt'),
    "software": os.path.join(PRE_TRAINED_VECTOR_PATH, 'software_prd.pt'),
}
EXTRA_PRD_VECTORS_MAP = {
    "imdb": os.path.join(PRE_TRAINED_VECTOR_PATH, 'imdb-embedding-200d.txt'),
    "yelp_13": os.path.join(PRE_TRAINED_VECTOR_PATH, 'yelp-2013-embedding-200d.txt'),
    "yelp_14": os.path.join(PRE_TRAINED_VECTOR_PATH, 'yelp-2014-embedding-200d.txt')
}

DATASET_MAP = {
    # "imdb": IMDB,
    "imdb": IMDBHierarchical,
    "yelp_13": YELP13Hierarchical,
    "yelp_14": YELP14Hierarchical,
    "digital": DIGITALHierarchical,
    "industrial": IndustrialHierarchical,
    "software": SoftwareHierarchical,
}

BASELINES = {
    'lstm': LSTM,
}

DATASET_MAP_LSTM = {
    "imdb": IMDB,
    "yelp_13": YELP13,
    "yelp_14": YELP14,
    "digital": DIGITAL,
    "industrial": Industrial,
    "software": Software,
}
