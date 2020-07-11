import encoding
import torchsummary
model = encoding.models.backbone.resnet50(pretrained=True)
torchsummary.summary(model, (3, 224, 224), device="cpu")