#!/usr/bin/env python3

import requests
import tqdm

CHUNK_SIZE = 1 * 1024 * 1024


def download_file(source, destination, size=None):
    if size is None:
        size = 0
    req = requests.get(source, stream=True)
    with open(destination, 'wb') as archive:
        for chunk in tqdm.tqdm(
            req.iter_content(chunk_size=CHUNK_SIZE),
            total=size // CHUNK_SIZE,
            leave=False,
        ):
            if chunk:
                archive.write(chunk)


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

def custom_get_task(self, task_description):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/6b741028f0812ba73da58acf7206d93654dde97a/learn2learn/data/task_dataset.pyx#L136)
    Given a task description, creates the corresponding batch of data.

    Returns:
        :all_data: [(data.x, data.remaped_y, data.real_y), (..), ...]
    """
    all_data = []
    # [(data, real_cls_idx)]
    for data_description in task_description:
        data = data_description.index
        for transform in data_description.transforms:
            data = transform(data)
        all_data.append((data, 
                        int(self.dataset.dataset.indices_to_labels[data_description.index])
                       ))
    return self.task_collate(all_data)

def custom_get_task_attn(self, task_description):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/6b741028f0812ba73da58acf7206d93654dde97a/learn2learn/data/task_dataset.pyx#L136)
    Given a task description, creates the corresponding batch of data.

    Returns:
        :all_data: [(data.x, data.remaped_y, data.real_y), (..), ...]
    """
    all_data = []
    # [(data, real_cls_idx)]
    for data_description in task_description:
        data = data_description.index
        for transform in data_description.transforms:
            data = transform(data)
        all_data.append((data, 
                        int(self.dataset.indices_to_labels[data_description.index])
                       ))
    return self.task_collate(all_data)

def get_classnames(dataset, taskset, use_task_gen=True):
    """
    Utility function to get dict of class labels and names
    
    Args:
        dataset: name of dataset ('mini-imagenet')
        taskset: (l2l.data.Taskset)
        use_task_gen: use task generator settings.
    Return:
        classes: (dict [cls_synset]: cls_labels)
        lbl_name: (dict [cls_label]: cls_name)
    """
    if dataset == 'mini-imagenet':
        imagenet_cls_names = {}
        with open("/home/user/TaskAttentionTG/src/datasets/imagenet-cls-feats/LOC_synset_mapping.txt", 'r') as file:
            for l in file:
                synset = l.split(' ')[0]
                names = ' '.join(l.split(' ')[1:]).replace('\n', '').split(', ')
                imagenet_cls_names.update({synset: names})

        lbl_name = {}
        if use_task_gen:
            classes = taskset.dataset.dataset.dataset.class_idx
        else:
            classes = taskset.dataset.dataset.class_idx
        # classes  = [synsets]: label
        for cls, label in classes.items():
            if label in taskset.dataset.labels:
                lbl_name.update({label: imagenet_cls_names[cls]})

        classes = {k:v for k, v in classes.items() if v in taskset.dataset.labels}

        return lbl_name, classes

    elif dataset == 'tiered-imagenet':
        imagenet_cls_names = {}
        with open("./datasets/imagenet-cls-feats/LOC_synset_mapping.txt", 'r') as file:
            for l in file:
                synset = l.split(' ')[0]
                names = ' '.join(l.split(' ')[1:]).replace('\n', '').split(', ')
                imagenet_cls_names.update({synset: ", ".join(names)})

        tiered_imagenet_path = os.path.join(taskset.dataset.dataset.dataset.root, 'tiered-imagenet')
        labels_path = os.path.join(tiered_imagenet_path, 'train' + '_labels.pkl')
        with open(labels_path, 'rb') as labels_file:
            labels = pickle.load(labels_file)
            
        tiered_imagenet_cls = {}
        # [synset: label]
        for synset, names in imagenet_cls_names.items():
            if names in labels['label_specific_str']:
                # two cranes in imagenet and only one is present in tiered-imagenet
                if names == 'crane' and synset == 'n03126707':
                    continue
                tiered_imagenet_cls.update({synset: labels['label_specific_str'].index(names)})

        lbl_name = {}
        for cls, label in tiered_imagenet_cls.items():
            if label in taskset.dataset.labels:
                lbl_name.update({label: imagenet_cls_names[cls].split(', ')})

        classes = {k:v for k, v in tiered_imagenet_cls.items() if v in taskset.dataset.labels}

        return lbl_name, classes

    else:
        raise Exception('Invalid dataset with task-generator')

def get_classification_data(dataset_name, dataset, test_size=0.2, batch_size=64, train_transform=None):
    "Utility function to split a torch.utils.data.Dataset into train and valid set"
    image_size=84
    train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(92),
            transforms.RandomResizedCrop(88),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[120.39586422/255.0, 115.59361427/255.0, 104.54012653/255.0],
                std=[70.68188272/255.0, 68.27635443/255.0, 72.54505529/255.0],
    )])
    test_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(92),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[120.39586422/255.0, 115.59361427/255.0, 104.54012653/255.0],
                std=[70.68188272/255.0, 68.27635443/255.0, 72.54505529/255.0],
    )])

    x, y = [], []
    for idx in range(len(dataset)):
        data = dataset[idx]
        x.append(data[0])
        y.append(data[1])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y)
    x_train = [train_transforms(x) for x in x_train]
    x_test = [test_transforms(x) for x in x_test]
    x_train_tensor = torch.stack(x_train)
    y_train_tensor = torch.LongTensor(y_train)
    x_test_tensor  = torch.stack(x_test)
    y_test_tensor  = torch.LongTensor(y_test)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset  = TensorDataset(x_test_tensor, y_test_tensor)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4), DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
