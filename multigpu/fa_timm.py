from fastai.vision.all import *
from fastai.distributed import *
from fastai.vision.models.xresnet import *
import timm

path = '.fastai/data/imagewoof2-320/'


dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    splitter=GrandparentSplitter(valid_name='val'),
    get_items=get_image_files, get_y=parent_label,
    item_tfms=[RandomResizedCrop(256), FlipItem(0.5)],
    batch_tfms=Normalize.from_stats(*imagenet_stats)
).dataloaders(path, path=path, bs=64)

timm_model = timm.create_model(model_name='tf_efficientnet_b4_ns',num_classes=10)
learn = Learner(dls,timm_model, metrics=[accuracy,top_k_accuracy]).to_fp16()
with learn.distrib_ctx(): 
    learn.fit_flat_cos(2, 1e-3, cbs=MixUp(0.1))