from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import random
import json

CATEGORIES = {
    'dog': 0,
    'elephant': 1,
    'giraffe': 2,
    'guitar': 3,
    'horse': 4,
    'house': 5,
    'person': 6,
}

DOMAINS = {
    'art_painting': 0,
    'cartoon': 1,
    'sketch': 2,
    'photo': 3,
}

DESCRIPTORS = {
    0: 'level of details',
    1: 'edges',
    2: 'color saturation',
    3: 'color shades',
    4: 'background',
    5: 'single instance',
    6: 'text',
    7: 'texture',
    8: 'perspective',
    9: 'realistic',
    10: 'unrealistic',
}

def assign_domain_labels(target_domain):
    if target_domain != 'photo':
        current = DOMAINS[target_domain]
        DOMAINS['photo'] = current
        DOMAINS[target_domain] = 3


class PACSDatasetBaseline(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, y = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y

def read_lines(data_path, domain_name):
    examples = {}
    
    with open(f'./Vision-Language-AML/{data_path}/{domain_name}.txt') as f:
        lines = f.readlines()

    for line in lines: 
        main_folder = './Vision-Language-AML/'
        local_folder = '/kfold/'
        line = line.strip().split()[0].split('/')
        category_name = line[3]
        category_idx = CATEGORIES[category_name]
        image_name = line[4]
        # image_path_zzz = f'./Vision-Language-AML/{data_path}/{domain_name}/{category_name}/{image_name}'
        # print(f'{line}, image_path_zzz {image_path_zzz}, image_name {image_name}, category_name {category_name}, category_idx {category_idx}')
        image_path = f'{main_folder}{data_path}{local_folder}{domain_name}/{category_name}/{image_name}'
        if category_idx not in examples.keys():
            examples[category_idx] = [image_path]
        else:
            examples[category_idx].append(image_path)

    return examples

def read_lines_DG(data_path, domains):
    examples = {}
    image_domain = {}
    
    for domain in domains:
        with open(f'./Vision-Language-AML/{data_path}/{domain}.txt') as f:
            lines = f.readlines()

        for line in lines: 
            main_folder = './Vision-Language-AML/'
            local_folder = '/kfold/'
            line = line.strip().split()[0].split('/')
            category_name = line[3]
            category_idx = CATEGORIES[category_name]
            image_name = line[4]
            image_path = f'{main_folder}{data_path}{local_folder}{domain}/{category_name}/{image_name}'
            image_domain[image_path] = DOMAINS[domain]  
            if category_idx not in examples.keys():
                examples[category_idx] = [image_path]
            else:
                examples[category_idx].append(image_path)

    return examples, image_domain

def build_splits_baseline(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    if opt['dom_gen'] == 'False':
        source_examples = read_lines(opt['data_path'], source_domain)
    else:
        choices = ['art_painting', 'cartoon', 'sketch', 'photo']
        source_examples, _ = read_lines_DG(opt['data_path'], [c for c in choices if c != target_domain])

    target_examples = read_lines(opt['data_path'], target_domain)

    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length = source_total_examples * 0.2 # 20% of the training split used for validation

    train_examples = []
    val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        split_idx = round(source_category_ratios[category_idx] * val_split_length)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                train_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
            else:
                val_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            test_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetBaseline(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetBaseline(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader


class PACSDatasetDomainDisentangle(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, y, domain = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y, domain

def build_splits_domain_disentangle(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    if opt['dom_gen'] == 'False':
        source_examples = read_lines(opt['data_path'], source_domain)
        target_examples = read_lines(opt['data_path'], target_domain)
    else:   
        choices = ['art_painting', 'cartoon', 'sketch', 'photo']
        assign_domain_labels(target_domain)
        source_examples, image_domain_s = read_lines_DG(opt['data_path'], [c for c in choices if c != target_domain])
        target_examples, _ = read_lines_DG(opt['data_path'], [target_domain])
    
    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    source_val_split_length = source_total_examples * 0.2 # 20% of the training split used for validation

    # Compute ratios of examples for each category
    target_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in target_examples.items()}
    target_total_examples = sum(target_category_ratios.values())
    target_category_ratios = {category_idx: c / target_total_examples for category_idx, c in target_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    target_val_split_length = target_total_examples * 0.2 # 20% of the training split used for validation

    train_examples_source = []
    val_examples_source = []
    train_examples_target = []
    val_examples_target = []
    test_examples = []
    
    for category, examples_list in source_examples.items():
        split_idx = round(source_category_ratios[category] * source_val_split_length)
        for i, example in enumerate(examples_list):
            if opt['dom_gen'] == 'False': 
                train_examples_source.append([example, category, 0]) if i>split_idx else val_examples_source.append([example, category, 0])
            else:
                train_examples_source.append([example, category, image_domain_s[example]]) if i>split_idx else val_examples_source.append([example, category, image_domain_s[example]])
    
    for category, examples_list in target_examples.items():
        split_idx = round(target_category_ratios[category] * target_val_split_length)
        for i, example in enumerate(examples_list):
            if opt['dom_gen'] == 'False': 
                test_examples.append([example, category, 1])
                train_examples_target.append([example, 42, 1]) if i>split_idx else val_examples_target.append([example, category, 1])
            else:   #se sono in dom gen passo anche dominio
                test_examples.append([example, category, 1])
    
    if opt['dom_gen'] == 'False':
        train_examples = train_examples_source + train_examples_target
        val_examples = val_examples_source + val_examples_target
    else:
        train_examples = train_examples_source
        val_examples = val_examples_source

    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders 
    train_loader = DataLoader(PACSDatasetDomainDisentangle(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetDomainDisentangle(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    test_loader = DataLoader(PACSDatasetDomainDisentangle(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader

class PACSDatasetClipDisentangle(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, desc, y, domain = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, desc, y, domain

class PACSDatasetClipPreTraining(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, desc = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, desc

def getDomain(path):
    return path.split('/')[5]

def getCategory(path):
    return path.split('/')[6]   

def pre_proc(vet):
    desc = ''
    for i, x in enumerate(vet):
        if i < 11:
            desc = desc + DESCRIPTORS[i] + ": " + x + "; " 
        else:
            desc = desc + x + "; "
    if len(desc) > 300:     # Truncate at 300
        desc = desc[0:300]    
    return desc

def readJSON(domains):
    '''
    return a dictionary with image paths as keys and image descriptions as values
    '''

    with open("./Vision-Language-AML/data/LabeledPACS/all_descriptions.json") as file:
        #print(file.read())
        data = json.loads(file.read())

        return {("./Vision-Language-AML/data/PACS/kfold/"+i['image_name']): pre_proc(i['descriptions']) for i in data if i['image_name'].split('/')[0] in domains}      


def build_splits_clip_disentangle(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    if opt['dom_gen'] == 'False':
        source_examples_dict = readJSON([source_domain])
        target_examples_dict = readJSON([target_domain])
    else:   
        choices = ['art_painting', 'cartoon', 'sketch', 'photo']
        assign_domain_labels(target_domain)
        source_examples_dict = readJSON([c for c in choices if c != target_domain])
        target_examples_dict = readJSON([target_domain])

    if opt['clip_pretrained'] == 'False':
        clip_examples = readJSON(['art_painting', 'cartoon', 'sketch', 'photo'])
        train_clip = [[img_path, desc] for img_path, desc in clip_examples.items()]
        
    '''
    create dict with category as key and list of (path, description) as value
    '''
    source_examples = dict()
    for k,v in source_examples_dict.items():
        if CATEGORIES[getCategory(k)] in source_examples:
            source_examples[CATEGORIES[getCategory(k)]].append((k, v))
        else:
            source_examples[CATEGORIES[getCategory(k)]] = list()
            source_examples[CATEGORIES[getCategory(k)]].append((k,v))

    target_examples = dict()
    for k,v in target_examples_dict.items():
        if CATEGORIES[getCategory(k)] in target_examples:
            target_examples[CATEGORIES[getCategory(k)]].append((k, v))
        else:
            target_examples[CATEGORIES[getCategory(k)]] = list()
            target_examples[CATEGORIES[getCategory(k)]].append((k,v))
                    
    
   # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    source_val_split_length = source_total_examples * 0.2 # 20% of the training split used for validation

    # Compute ratios of examples for each category
    target_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in target_examples.items()}
    target_total_examples = sum(target_category_ratios.values())
    target_category_ratios = {category_idx: c / target_total_examples for category_idx, c in target_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    target_val_split_length = target_total_examples * 0.2 # 20% of the training split used for validation

    train_examples_source = []
    val_examples_source = []
    train_examples_target = []
    val_examples_target = []
    test_examples = []
    
    for category, examples_list in source_examples.items():
        split_idx = round(source_category_ratios[category] * source_val_split_length)
        for i, example in enumerate(examples_list):
            if opt['dom_gen'] == 'False': 
                train_examples_source.append([example[0], example[1], category, 0]) if i>split_idx else val_examples_source.append([example[0], example[1], category, 0])
            else:
                train_examples_source.append([example[0], example[1], category, DOMAINS[getDomain(example[0])]]) if i>split_idx else val_examples_source.append([example[0], example[1], category, DOMAINS[getDomain(example[0])]])
    
    for category, examples_list in target_examples.items():
        split_idx = round(target_category_ratios[category] * target_val_split_length)
        for i, example in enumerate(examples_list):
            if opt['dom_gen'] == 'False': 
                test_examples.append([example[0], example[1], category, 1])
                train_examples_target.append([example[0], example[1], 42, 1]) if i>split_idx else val_examples_target.append([example[0], example[1], category, 1])
            else:   
                test_examples.append([example[0], example[1], category, DOMAINS[getDomain(example[0])]])
                
    if opt['dom_gen'] == 'False':            
        train_examples = train_examples_source + train_examples_target
        val_examples = val_examples_source + val_examples_target
    else:
        train_examples = train_examples_source
        val_examples = val_examples_source

    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders 
    if opt['clip_pretrained'] == 'False':
        train_clip_loader = DataLoader(PACSDatasetClipPreTraining(train_clip, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
        train_loader = DataLoader(PACSDatasetClipDisentangle(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
        val_loader = DataLoader(PACSDatasetClipDisentangle(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
        test_loader = DataLoader(PACSDatasetClipDisentangle(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
        
        return train_loader, val_loader, test_loader, train_clip_loader
    else:
        train_loader = DataLoader(PACSDatasetClipDisentangle(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
        val_loader = DataLoader(PACSDatasetClipDisentangle(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
        test_loader = DataLoader(PACSDatasetClipDisentangle(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
        
        return train_loader, val_loader, test_loader
