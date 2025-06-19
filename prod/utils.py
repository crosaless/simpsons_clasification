import torch
import torch.nn as nn
from torchvision.models import densenet121

# Lista de clases (labels)
CLASSES = [
    'abraham_grampa_simpson',
    'agnes_skinner',
    'apu_nahasapeemapetilon',
    'bart_simpson',
    'barney_gumble',
    'carl_carlson',
    'charles_montgomery_burns',
    'chief_wiggum',
    'cletus_spuckler',
    'comic_book_guy',
    'disco_stu',
    'edna_krabappel',
    'fat_tony',
    'gil',
    'groundskeeper_willie',
    'homer_simpson',
    'kent_brockman',
    'krusty_the_clown',
    'lenny_leonard',
    'lisa_simpson',
    'lionel_hutz',
    'maggie_simpson',
    'marge_simpson',
    'martin_prince',
    'mayor_quimby',
    'milhouse_van_houten',
    'miss_hoover',
    'moe_szyslak',
    'nelson_muntz',
    'ned_flanders',
    'otto_mann',
    'patty_bouvier',
    'principal_skinner',
    'professor_john_frink',
    'ralph_wiggum',
    'rainier_wolfcastle',
    'selma_bouvier',
    'sideshow_bob',
    'sideshow_mel',
    'snake_jailbird',
    'troy_mcclure',
    'waylon_smithers'
]

# Diccionario índice 
idx_to_class = {i: label for i, label in enumerate(CLASSES)}

def load_model(path):
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model, idx_to_class

def predict_character(model, img_tensor, idx_to_class):
    outputs = model(img_tensor)
    _, predicted_idx = torch.max(outputs, 1)
    return idx_to_class.get(predicted_idx.item(), "Desconocido")
