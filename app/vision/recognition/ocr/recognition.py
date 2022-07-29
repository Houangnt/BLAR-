import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image

from app.vision.recognition.ocr.network.utils import AttnLabelConverter
from app.vision.recognition.ocr.network.model import Model
from app.common.base import RecognitionBase


class TextRecognizer(RecognitionBase):
    def __init__(self, model_path, vocab='0123456789', nc=3, img_height=32, img_width=100, device='cpu'):
        super().__init__(recognition_type="barcode", version="tocr")
        self.device = torch.device('cuda' if (device == 'cuda' and torch.cuda.is_available()) else 'cpu')
        self.nc = nc
        self.batch_max_length = 10
        self.img_height, self.img_width = img_height, img_width
        self.character = vocab
        self.model_path = model_path

        self.decoder = AttnLabelConverter(self.character, self.device)
        self.model = Model(num_class=len(self.decoder.character), image_size=(self.img_height, self.img_width),
                           device=self.device)
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self._load_weights()
        self.model.eval()

        self.text_for_pred = torch.empty(1, self.batch_max_length + 1, dtype=torch.long).fill_(0).to(self.device)
        # self.length_for_pred = torch.empty(1, dtype=torch.int).fill_(self.batch_max_length).to(self.device)
        self.length_for_pred = torch.IntTensor([self.batch_max_length] * 1).to(device)

    def _preprocess(self, cv_img):
        img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img.astype(np.uint8))
        img = img.resize((self.img_width, self.img_height), Image.BICUBIC)
        img = transforms.ToTensor()(img)
        img.sub_(0.5).div_(0.5)
        img = img.to(self.device).unsqueeze(dim=0)
        return img

    def _recognize(self, imgs):
        result = []
        for img in imgs:
            image = self._preprocess(img)
            preds = self.model(image, self.text_for_pred, is_train=False)
            _, preds_index = preds.max(2)
            pred_str = self.decoder.decode(preds_index, self.length_for_pred)[0]
            # pred_max_prob, _ = F.softmax(preds, dim=2).max(dim=2)
            end_idx = pred_str.find('[s]')
            if end_idx == -1:
                pred_str += ' '
            text = pred_str[:end_idx]
            # prob = pred_max_prob[0][:end_idx].cumprod(dim=0)[-1]
            result.append(text)
        return result

    def _load_weights(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
