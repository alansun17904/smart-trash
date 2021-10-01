import io
import json
import torch
import numpy as np
from PIL import Image
from meta_bilinear import ResNet
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = ResNet()
model.load_state_dict(torch.load('models/thanosnet.pth',
									map_location=device))
model.eval()

labels_dict = {
    0: 'Can',
    1: 'Landfill',
    2: 'Paper',
    3: 'Plastic',
    4: 'Tetrapak'
}

metadata = json.load(open('data/metadata.json'))

def transform_image(image):
	inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
	])
	image = Image.open(image).convert('RGB')

	return inference_transform(image).unsqueeze(0)

def predict_image(image: str, location: str) -> str:
	if location not in metadata.keys():
		meta = metadata.get('default')
	else:
		meta = metadata.get(location)

	# Load metadata
	meta_ = torch.tensor(meta).unsqueeze(0)
	meta_ = meta_.to(device)

	# Load and normalize image
	tensor = transform_image(image)
	input_img = tensor.to(device)
	output = model.forward(input_img, meta_)
	index = output.data.cpu().numpy().argmax()
	return labels_dict[index]
