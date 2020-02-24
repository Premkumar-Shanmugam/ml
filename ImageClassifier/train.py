import torch

from utils import get_args, get_device, get_loader, get_class_to_idx
from modeler import get_model, train_model, save_model

## Get the arguments
args = get_args('train')

## Get the data loaders
trainloader = get_loader(set='train', data_dir=args.data_dir)
validloader = get_loader(set='valid', data_dir=args.data_dir)

## Get the model and the architecture
new_model = get_model(arch=args.arch,
                      l1units=args.l1_hidden_units,
                      l2units=args.l2_hidden_units)

## Get the device
device = get_device(args.gpu)

## Get the trained and validated the model
trained_validated_model = train_model(model=new_model,
                                            device=device,
                                            epochs=args.epochs,
                                            learn_rate=args.learn_rate,
                                            trainloader=trainloader,
                                            validloader=validloader,
                                            print_every=50)

## Get Class to Index mapping from train set
class_to_idx = get_class_to_idx(set='train', data_dir=args.data_dir)

## Save the trained and validaed model
save_model(arch=args.arch,
           model=trained_validated_model, 
           mapping=class_to_idx,
           save_dir=args.save_dir,
           state='trained_validated')
