from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import random

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

def read_lines_DG(data_path, domain_name):
    examples = {}

    for domain in domain_name:
        with open(f'./Vision-Language-AML/{data_path}/{domain}.txt') as f:
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
            image_path = f'{main_folder}{data_path}{local_folder}{domain}/{category_name}/{image_name}'
            if category_idx not in examples.keys():
                examples[category_idx] = [(image_path, DOMAINS[domain])]
            else:
                examples[category_idx].append((image_path, DOMAINS[domain]))
    
    return examples

def build_splits_baseline(opt, mode=None):
    
    if mode == None:
        source_domain = 'art_painting'
        target_domain = opt['target_domain']
        source_examples = read_lines(opt['data_path'], source_domain)
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

    elif mode == 'DG':
        choices=['art_painting', 'cartoon', 'sketch', 'photo']
        target_domain = opt['target_domain']
        source_domains = [x for x in choices if x!=target_domain]
        source_examples = read_lines_DG(opt['data_path'], source_domains)
        target_examples = read_lines_DG(opt['data_path'], target_domain)

        # Compute ratios of examples for each category
        source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list[0] in source_examples.items()}
        source_total_examples = sum(source_category_ratios.values())
        source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

        # Build splits - we train only on the source domain (Art Painting)
        val_split_length = source_total_examples * 0.2 # 20% of the training split used for validation

        train_examples = []
        val_examples = []
        test_examples = []

        for category_idx, examples_list, _ in source_examples.items():
            split_idx = round(source_category_ratios[category_idx] * val_split_length)
            for i, example in enumerate(examples_list):
                if i > split_idx:
                    train_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
                else:
                    val_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
        
        for category_idx, examples_list, _ in target_examples.items():
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

def build_splits_domain_disentangle(opt, mode=None):

    if mode == None:
        source_domain = 'art_painting'
        target_domain = opt['target_domain']
        source_examples = read_lines(opt['data_path'], source_domain)
        target_examples = read_lines(opt['data_path'], target_domain)

        train_examples_1 = []
        train_examples_2 = []
        test_examples = []

        for category, example_list in source_examples.items():
            for example in example_list:
                train_examples_1.append([example, category, 0])
        
        for category, example_list in target_examples.items():
            for example in example_list:
                train_examples_2.append([example, category, 1])
                test_examples.append([example, category, 1])

    elif mode == 'DG':
        choices=['art_painting', 'cartoon', 'sketch', 'photo']
        target_domain = opt['target_domain']
        source_domains = [x for x in choices if x!=target_domain]
        source_examples = read_lines_DG(opt['data_path'], source_domains, mode)
        target_examples = read_lines_DG(opt['data_path'], target_domain, mode)

        train_examples_1 = []
        train_examples_2 = []
        test_examples = []

        for category, example_list, domain in source_examples.items():
            for example in example_list:
                train_examples_1.append([example, category, domain])
        
        for category, example_list, domain in target_examples.items():
            for example in example_list:
                train_examples_2.append([example, category, domain])
                test_examples.append([example, category, domain])

    # Train and Val from source -> both domain encoder + domain clf and category encoder + category clf
    train_examples_source = train_examples_1[0:round(0.8*len(train_examples_1))]
    val_examples_both = train_examples_1[round(0.8*len(train_examples_1)):]

    # Train and Val from domain -> only domain encoder + domain clf
    train_examples_domain = train_examples_2[0:round(0.8*len(train_examples_2))]
    val_examples_domain = train_examples_2[round(0.8*len(train_examples_2)):]

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
    train_loader_1 = DataLoader(PACSDatasetDomainDisentangle(train_examples_source, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True) 
    train_loader_2 = DataLoader(PACSDatasetDomainDisentangle(train_examples_domain, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader_1 = DataLoader(PACSDatasetDomainDisentangle(val_examples_both, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    val_loader_2 = DataLoader(PACSDatasetDomainDisentangle(val_examples_domain, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetDomainDisentangle(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader_1, train_loader_2, val_loader_1, val_loader_2, test_loader

class PACSDatasetClipDisentangle(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, y, domain = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y, domain

def build_splits_clip_disentangle(opt, mode=None):
    pass
