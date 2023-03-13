# %%
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, VideoMAEConfig, VideoMAEFeatureExtractor
import torch
import torch.nn as nn

# %%
class HuggingFaceModels(nn.Module):
    '''
    the module zoo from the PytorchVideo lib.

    Args:
        nn (_type_): 
    '''

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_class_num = hparams.model_class_num

        # self.transfor_learning = hparams.transfor_learning

    def make_video_mae(self):

        # model_ckpt = "MCG-NJU/videomae-base"
        model_ckpt = "MCG-NJU/videomae-base-finetuned-kinetics"
        image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
        feature_extractor = VideoMAEFeatureExtractor.from_pretrained(model_ckpt)
        config = VideoMAEConfig.from_pretrained(model_ckpt)

        model = VideoMAEForVideoClassification.from_pretrained(
            model_ckpt,
            ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )

        # change the knetics-400 output 400 to model class num
        model.classifier = nn.Linear(768, self.model_class_num, bias=True)

        return model, image_processor, feature_extractor
    
# %%

# class opt:
#     model_class_num = 2

# hugging_face_model = HuggingFaceModels(opt)


# # %%
# model, image_processor = hugging_face_model.make_video_mae()
# %%
