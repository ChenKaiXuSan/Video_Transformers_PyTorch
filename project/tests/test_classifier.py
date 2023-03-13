'''
just for debug
'''

# %%
import torch 
import os 
# %cd /workspace/Walk_Video_PyTorch/project

# %%
from models.pytorchvideo_models import WalkVideoClassificationLightningModule
from pytorch_lightning import Trainer

from IPython.display import clear_output

clear_output()

# %%
from pytorch_lightning import seed_everything

seed_everything(42, workers=True)

# %%
from main import get_parameters

opt, unknown = get_parameters()
opt.num_workers = 8
opt.batch_size = 4
opt.gpu_num = 1

opt.version = '1114_1_16'
opt.model = "resnet"
opt.model_depth = 50

opt.clip_duration = 1
opt.uniform_temporal_subsample_num = 16
opt.version = opt.version + '_' + opt.model + '_depth' + str(opt.model_depth)


# %%
from utils.utils import get_ckpt_path

model = WalkVideoClassificationLightningModule(opt)

# get last ckpt path
# ckpt_path = get_ckpt_path(opt)
ckpt_path = '/workspace/Walk_Video_PyTorch/logs/resnet/1114_1_16_resnet_depth50/checkpoints/epoch=2-val_loss=3.56-val_acc=0.5034.ckpt'

# model = WalkVideoClassificationLightningModule.load_from_checkpoint(ckpt_path)
model = model.load_from_checkpoint(ckpt_path)

model.eval()

# clear_output()
print(ckpt_path)


# %%
from dataloader.data_loader import WalkDataModule
from pytorch_lightning import loggers as pl_loggers 

# load test dataset 
module = WalkDataModule(opt)
module.setup()
test_data = module.test_dataloader()

# for the tensorboard
tb_logger = pl_loggers.TensorBoardLogger(save_dir="/workspace/Walk_Video_PyTorch/project/tests/logs", name=opt.model, version=opt.version)

trainer = Trainer(
    accelerator="auto",
    devices=1,
    gpus=opt.gpu_num,
    logger=tb_logger,
    max_epochs=5,
    # deterministic=True,
)

final_acc = trainer.test(dataloaders=module, model=model)

# %%
# trainer.validate(dataloaders=module, model=model, ckpt_path=ckpt_path)

# %%
input_data = next(iter(test_data))


# %%
input_data['video'].shape

input_data["video_name"]

# input_data['video'][0][0][0][0]

# %%
input_data['label']

# %%
preds = model(input_data['video'])

# %%
post_act = torch.nn.Softmax(dim=1)
preds = post_act(preds)
pred_classes = preds.topk(k=1).indices

# %%
pred_classes

# %%
from torch import softmax


pred_chagne = softmax(preds, dim=-1)
pred_chagne

# %%
from utils.metrics import get_Accuracy

accuracy = get_Accuracy()

accuracy(pred_chagne, input_data['label'])

# %%
classname = {}

classname[0] = 'asd'
classname[1] = 'asd_not'

# %%
real_calss = []

for i in input_data['label'].tolist():
    real_calss.append(classname[i])

# %%
real_calss

# %%
# pred_class_names = []
# for num in range(opt._BATCH_SIZE):
#     for i in pred_classes[i]:
#         pred_class_names.append(classname[int(i)])


pred_class_names = [classname[int(i)] for i in pred_classes]
print("Predicted labels: %s" % ", ".join(pred_class_names))
print("real label: %s" % ",".join(real_calss))

# %%
pred_class_names == real_calss

# %%
result = []

for i in range(len(real_calss)):
    if pred_class_names[i] == real_calss[i]:
        result.append("true")
    else:
        result.append("false")

result


