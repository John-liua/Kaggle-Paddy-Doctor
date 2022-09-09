from core.loss import *
import torchvision.models as models
from core.utils import *
from core.seed import *

model = models.resnet50(pretrained=True)
print(model)