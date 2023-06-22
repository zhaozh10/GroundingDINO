from dataset.data import get_transforms, get_datasets_as_df, get_datasets
from dataset.object_dataset import ObjectSet
from dataset.det_text_dataset import DetTextSet
from dataset.tokenizer import get_tokenized_datasets, get_tokenizer
from datasets import load_from_disk
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)
config_file_path='./run_config.txt'



train_transforms = get_transforms("train")
# val_transforms = get_transforms("val")

# tokenizer=get_tokenizer()

# load tokenized dataset
tokenized_train = load_from_disk("tokenized_train")
tokenized_val = load_from_disk("tokenized_val")
tokenized_test=load_from_disk('tokenized_test')
tokenized_test_2=load_from_disk('tokenized_test_2')
train_dataset = DetTextSet("train", tokenized_train, train_transforms, log)
# train_dataset = ObjectSet(datasets_as_df["train"], train_transforms)
ret=train_dataset.__getitem__(index=10)
print("hi")
