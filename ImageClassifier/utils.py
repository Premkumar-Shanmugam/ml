import argparse
import torch
from torchvision import datasets, transforms

from PIL import Image
import numpy as np
import json

## Function to parse command-line arguments
def get_args(mode='train'):
    ## Add arguments
    if (mode == 'train'):
        parser = argparse.ArgumentParser(description='AI Model Trainer.')
        parser.add_argument('data_dir', action='store', help='Directory with input images.')
        parser.add_argument('--arch', action='store', default='vgg11', choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'], help='Pre-trained model to use.')
        parser.add_argument('--l1_hidden_units', action='store', default=4096, type=int, help='Hidden units in Layer-1.')
        parser.add_argument('--l2_hidden_units', action='store', default=1024, type=int, help='Hidden units in Layer-2.')
        parser.add_argument('--save_dir', action='store', default='./', help='Directory to save the model.')
        parser.add_argument('--learn_rate', action='store', default=0.001, type=float, help='Learning Rate to be used by the model to train.')
        parser.add_argument('--epochs', action='store', default=4, type=int, help='Number of batches of images to train.')
    elif (mode == 'test'):
        parser = argparse.ArgumentParser(description='AI Model Tester.')
        parser.add_argument('data_dir', action='store', help='Directory with input images.')
        parser.add_argument('--model_dir', action='store', default='./', help='Directory to pick the model from.')
    elif (mode == 'predict'):
        parser = argparse.ArgumentParser(description='AI Model Predictor.')
        parser.add_argument('image_path', action='store', help='Image to predict.')
        parser.add_argument('--model_dir', action='store', default='./', help='Directory to pick the model from.')
        parser.add_argument('--topk', action='store', default=1, choices=range(1, 103), type=int, help='Top K predictions to make.')
        parser.add_argument('--cat_names', action='store', default='cat_to_name.json', help='Mapping of classes to names.')
    else:
        exit('System Error: 1 | Invalid \'Mode\' ' + mode + ' in retrieving arguments')

    parser.add_argument('--gpu', action="store_true", default=False, help='Use a GPU')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    ## Parse arguments
    args = parser.parse_args()
    
    ## Print arguments
    print('Parameters used:')
    for arg in vars(args):
        print('  ', arg, ': ', getattr(args,arg))
    
    ## Return arguments
    return args

## Function to get the device (GPU or CPU) based on user's request and the device availability
def get_device(gpu=False):
    device = torch.device('cpu')
    if gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print('GPU not available; using CPU...')
    return device

## Function to load data
def get_loader(set='train', data_dir='.'):
    data_dir = data_dir + '/' + set
    ## Define transforms for the training, validation, and testing sets
    if set == 'train':
        data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])
                                             ])
        shuffle_data = True
    elif (set == 'valid') | (set == 'test'):
        data_transforms = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])
                                              ])
        shuffle_data = False
    else:
        exit('System Error: 2 | Invalid \'Set\' ' + set + ' in loader arguments')

    ## Load the dataset with ImageFolder
    dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

    ## Using the image dataset and the trainforms, define the data loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=shuffle_data)
    
    ## Return the data loader
    return dataloader

## Function to get class to idx mapping from dataset for storing in the checkpoint
def get_class_to_idx(set='train', data_dir='.'):
    if set in ('train', 'valid', 'test'):
        dataset = datasets.ImageFolder(data_dir + '/' + set)
        return dataset.class_to_idx
    else:
        print('Warning: 1 | Invalid \'Set\' ' + set + ' in class_to_idx. No mapping returned.')
        return None

## Function to process image as np array
def process_image(image_path):

    ## Process a PIL image for use in a PyTorch model
    img = Image.open(image_path)
    
    ## Resize image to keep shorter side as 256
    img_width, img_height = img.size
    img_aspect_ratio = img_width / img_height
    short_side = 256
    
    if img_height > img_width:
        new_size = (short_side, short_side / img_aspect_ratio)
    else:
        new_size = (short_side * img_aspect_ratio, short_side)
    
    img.thumbnail(new_size)
    
    ## Crop out the center 224 x 224
    width, height = img.size
    crop_out = 224

    left = (width - crop_out) / 2
    top = (height - crop_out) / 2
    right = (width + crop_out) / 2
    bottom = (height + crop_out) / 2

    img = img.crop((left, top, right, bottom))
    
    ## Convert color channels from 0-255 to 0-1
    np_image = np.array(img) / 255
    
    ## Normalize: Subtract mean and divide by std. deviation
    mean = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std_dev
    
    ## Move color channel from 3rd dimension to 1st dimension
    np_image = np_image.transpose((2,0,1))
    
    ## return processes image
    return np_image

## Function to get class names from json
def get_class_names(classes=None,
                   cat_names='cat_to_name.json'):
    ## Open and read the json
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)

    ## Get names of the classes
    class_names = [cat_to_name[i] for i in classes]
    
    return class_names
