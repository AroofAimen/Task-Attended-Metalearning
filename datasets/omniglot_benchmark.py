import random
import learn2learn as l2l

from torchvision import transforms
from PIL.Image import LANCZOS
from .memorization import XRemapLabels,XNWays

def omniglot_tasksets(
    train_ways,
    train_samples,
    test_ways,
    test_samples,
    root,
    memorization=False,
    **kwargs
):
    """
    Benchmark definition for Omniglot.
    """
    data_transforms = transforms.Compose([
        transforms.Resize(28, interpolation=LANCZOS),
        transforms.ToTensor(),
        lambda x: 1.0 - x,
    ])
    omniglot = l2l.vision.datasets.FullOmniglot(
        root=root,
        transform=data_transforms,
        download=True,
    )
    dataset = l2l.data.MetaDataset(omniglot)
    train_dataset = dataset
    validation_datatset = dataset
    test_dataset = dataset

    classes = list(range(1623))
    random.shuffle(classes)

    if(memorization):
        train_transforms = [

            XNWays(dataset, train_ways,filter_labels=classes[:980]),
            l2l.data.transforms.KShots(dataset, train_samples),
            # l2l.data.transforms.FusedNWaysKShots(dataset,
            #                                     n=train_ways,
            #                                     k=train_samples,
            #                                     filter_labels=classes[:980]),
            l2l.data.transforms.LoadData(dataset),
            XRemapLabels(dataset),
            l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
        ]
        validation_transforms = [
            XNWays(dataset, test_ways,filter_labels=classes[980:1200]),
            l2l.data.transforms.KShots(dataset, test_samples),
            # l2l.data.transforms.FusedNWaysKShots(dataset,
            #                                     n=test_ways,
            #                                     k=test_samples,
            #                                     filter_labels=classes[980:1200]),
            l2l.data.transforms.LoadData(dataset),
            XRemapLabels(dataset),
            l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
        ]
        test_transforms = [
            XNWays(dataset, test_ways,filter_labels=classes[1200:]),
            l2l.data.transforms.KShots(dataset, test_samples),
            # l2l.data.transforms.FusedNWaysKShots(dataset,
            #                                     n=test_ways,
            #                                     k=test_samples,
            #                                     filter_labels=classes[1200:]),
            l2l.data.transforms.LoadData(dataset),
            XRemapLabels(dataset),
            l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
        ]

    else:
        train_transforms = [
            l2l.data.transforms.FusedNWaysKShots(dataset,
                                                n=train_ways,
                                                k=train_samples,
                                                filter_labels=classes[:980]),
            l2l.data.transforms.LoadData(dataset),
            l2l.data.transforms.RemapLabels(dataset),
            l2l.data.transforms.ConsecutiveLabels(dataset),
            l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
        ]
        validation_transforms = [
            l2l.data.transforms.FusedNWaysKShots(dataset,
                                                n=test_ways,
                                                k=test_samples,
                                                filter_labels=classes[980:1200]),
            l2l.data.transforms.LoadData(dataset),
            l2l.data.transforms.RemapLabels(dataset),
            l2l.data.transforms.ConsecutiveLabels(dataset),
            l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
        ]
        test_transforms = [
            l2l.data.transforms.FusedNWaysKShots(dataset,
                                                n=test_ways,
                                                k=test_samples,
                                                filter_labels=classes[1200:]),
            l2l.data.transforms.LoadData(dataset),
            l2l.data.transforms.RemapLabels(dataset),
            l2l.data.transforms.ConsecutiveLabels(dataset),
            l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
        ]

    _datasets = (train_dataset, validation_datatset, test_dataset)
    _transforms = (train_transforms, validation_transforms, test_transforms)
    return _datasets, _transforms
