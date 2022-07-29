import torch.nn as nn

from app.vision.recognition.ocr.network.transformation import TPS_SpatialTransformerNetwork
from app.vision.recognition.ocr.network.spin_transformation import SPIN
from app.vision.recognition.ocr.network.feature_extraction import VGGFeatureExtractor
from app.vision.recognition.ocr.network.prediction import Attention


class Model(nn.Module):

    def __init__(self, num_class, image_size, batch_max_length=9, device=None):
        super(Model, self).__init__()
        self.batch_max_length = batch_max_length
        self.image_size = image_size
        self.Transformation = TPS_SpatialTransformerNetwork(20, image_size, image_size, 3)
        # self.Transformation = SPIN(nc=3, k=6, device=device)

        self.FeatureExtraction = VGGFeatureExtractor(3, 512)
        self.FeatureExtraction_output = 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.Prediction = Attention(self.FeatureExtraction_output, 256, num_class, device=device)

    def forward(self, input_, text, is_train=True):
        input_ = self.Transformation(input_)
        visual_feature = self.FeatureExtraction(input_)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3).contiguous()
        prediction = self.Prediction(visual_feature, text, is_train, batch_max_length=self.batch_max_length)

        return prediction
