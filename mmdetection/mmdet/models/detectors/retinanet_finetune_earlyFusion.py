from mmdet.registry import MODELS
from .retinanet import RetinaNet


@MODELS.register_module()
class RetinaNetFineTune(RetinaNet):
    def __init__(self, **kwargs):
        super(RetinaNetFineTune, self).__init__(**kwargs)
        self._freeze_parameters()

    def _freeze_parameters(self):
        for name, para in self.named_parameters():
            if not ('backbone.stem' in name or 'bbox_head.retina_cls' in name):
                print(f'freezing {name}')
                para.requires_grad = False

