from dataset.data import get_transforms, CustomCollator
from dataset.det_text_dataset import DetTextSet
from dataset.tokenizer import get_tokenized_datasets, get_tokenizer
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)



train_transforms = get_transforms("train")
val_transforms = get_transforms("val")

tokenizer=get_tokenizer()

# load tokenized dataset
tokenized_train = load_from_disk("tokenized_train")
tokenized_val = load_from_disk("tokenized_val")
tokenized_test=load_from_disk('tokenized_test')
tokenized_test_2=load_from_disk('tokenized_test_2')
train_dataset = DetTextSet("train", tokenized_train, train_transforms)
val_dataset = DetTextSet("val", tokenized_val, val_transforms)
test_dataset=DetTextSet("test", tokenized_val, val_transforms)
# train_dataset = ObjectSet(datasets_as_df["train"], train_transforms)
# ret=train_dataset.__getitem__(index=10)


batchsize=64
train_loader = DataLoader(
        train_dataset,
        collate_fn=CustomCollator(tokenizer=tokenizer, is_val=False),
        batch_size=batchsize,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
val_loader = DataLoader(
        val_dataset,
        collate_fn=CustomCollator(tokenizer=tokenizer, is_val=True),
        batch_size=batchsize,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
test_loader = DataLoader(
        test_dataset,
        collate_fn=CustomCollator(tokenizer=tokenizer, is_val=True),
        batch_size=batchsize,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
for i,elem in enumerate(tqdm(train_loader)):
    pass
for i, elem in enumerate(tqdm(val_loader)):
    pass
for i, elem in enumerate(tqdm(test_loader)):
    pass
# ret=next(iter(data_loader))
print("hold")