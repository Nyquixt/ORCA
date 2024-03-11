from collections import OrderedDict
import json
import torch
import torch.nn.functional as F
from torchvision.datasets import EuroSAT
from torch.utils.data import DataLoader
import argparse
import clip
import torchmetrics
from tqdm import tqdm
from metrics import calc_fpr_aupr
from model import ConfidenceModel

# some misc 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='ViT-B/32')
parser.add_argument("--n-concepts", type=int, default=5)
parser.add_argument("--method", type=str, choices=['base', 'rank'])
args = parser.parse_args()

DATA_DIR = '<INSERT_FOLDER_NAME_HERE>'

# 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(args.model, device=device, jit=False)
model.eval()
model.requires_grad_(False)
model = model.float()

# load dataset and dataloader
# evaluate on the whole EuroSAT dataset
dataset = EuroSAT(root=DATA_DIR, transform=preprocess, download=True)
dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=16, pin_memory=True)

# load concepts
with open(f'descriptions/eurosat.json', 'rb') as infile:
    data = json.load(infile)
    classes = list(data.keys())

description_encodings = OrderedDict()

for k, v in data.items():
    desc = [f'a satellite image of {k}, {c}' for c in v[:args.n_concepts]]
    desc_tokens = clip.tokenize(desc).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        description_encodings[k] = F.normalize(model.encode_text(desc_tokens))

print(f"Zero-shot prediction on EuroSAT...")

all_confidences = []
corrects = []

conf_model = ConfidenceModel(args.method, model, description_encodings, 
                             args.n_concepts, len(classes))
accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=len(classes)).to(device)
for idx, batch in enumerate(tqdm(dataloader)):
    images, labels = batch

    images = images.to(device)
    labels = labels.to(device)
    one_hot_labels = F.one_hot(labels, num_classes=len(classes))
    
    # retrieve prediction and confidence score
    predictions, confidences, image_description_similarity = conf_model.predict(images)
    all_confidences.append(confidences.detach().cpu())
    corrects.append(predictions.cpu().eq(labels.cpu().data.view_as(predictions)))

    # accuracy
    acc = accuracy_metric(predictions.detach().cpu(), labels.detach().cpu())

all_confidences = torch.cat(all_confidences).numpy()
corrects = torch.cat(corrects).numpy()

# calculate metrics
print(f"Calculating metrics ...")

auroc, _, _, fpr_in_tpr_95, _ = calc_fpr_aupr(all_confidences, corrects)
print("AUROC {0:.2f}".format(auroc*100))
print('FPR@TPR95 {0:.2f}'.format(fpr_in_tpr_95*100))
print(f'Acc: {100 * accuracy_metric.compute().item()}')