from dataset.transforms import get_datasets_as_df,get_transforms
from dataset.object_dataset import ObjectSet
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)



datasets_as_df=get_datasets_as_df('./run_config.txt')
train_transforms = get_transforms("train")
val_transforms = get_transforms("val")

train_dataset = ObjectSet(datasets_as_df["train"], train_transforms)
train_dataset.__getitem__(index=10)