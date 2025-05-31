import torch
import zmq
from transformers import AutoModel, AutoProcessor
from torchvision import transforms
import pickle
from PIL import Image

class FeatureServer:
    def __init__(self, rank):
        self.rank = rank
        self.port = 5555 + rank

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.port}")

        # Load both models on GPU
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        self.dinov2.cuda(rank).eval()

        self.siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
        self.siglip.cuda(rank).eval()
        self.siglip_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

        # DINOv2 preprocessing
        self.dinov2_preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def process_image(self, image):
        assert type(image) == Image.Image
        inputs = self.dinov2_preprocess(image).unsqueeze(0).cuda(self.rank)

        with torch.no_grad():
            outputs = self.dinov2.forward_features(inputs)

        # Extract the last hidden state as features
        patch_features = outputs["x_norm_patchtokens"].squeeze(0)
        num_patches = patch_features.shape[0]
        h = w = int(num_patches**0.5)

        # Permute to [C, H, W] for interpolation
        patch_features_2d = patch_features.reshape(h, w, -1).permute(2, 0, 1)

        # Upsample to match original image patch dimensions
        resized_features = F.interpolate(
            patch_features_2d.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        return resized_features.squeeze().permute(1, 2, 0).cpu().numpy()

    def process_text(self, text):
        inputs = self.siglip_processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.cuda(self.rank) for k, v in inputs.items()}

        with torch.no_grad():
            text_embedding = self.siglip.get_text_features(**inputs)

        return text_embedding.cpu().squeeze().numpy()

    def run(self):
        while True:
            try:
                data = self.socket.recv()
                request = pickle.loads(data)

                if request['type'] == 'image':
                    result = self.process_image(request['data'])
                elif request['type'] == 'text':
                    result = self.process_text(request['data'])
                else:
                    raise ValueError(f"Unknown request type: {request['type']}")

                self.socket.send(pickle.dumps(result))
            except Exception as e:
                self.socket.send(pickle.dumps(e))

class FeatureClient:
    def __init__(self, rank):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{5555 + rank}")

    def extract_image_features(self, image_data):
        request = {'type': 'image', 'data': image_data}
        self.socket.send(pickle.dumps(request))
        data = self.socket.recv()
        result = pickle.loads(data)
        if isinstance(result, Exception):
            raise result
        return result

    def extract_text_features(self, text):
        request = {'type': 'text', 'data': text}
        self.socket.send(pickle.dumps(request))
        data = self.socket.recv()
        result = pickle.loads(data)
        if isinstance(result, Exception):
            raise result
        return result
