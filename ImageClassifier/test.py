from utils import get_args, get_device, get_loader
from modeler import load_model, test_model

## Get the arguments
args = get_args('test')

## Get the device
device = get_device(args.gpu)

## Get the model saved as checkpoint
trained_validated_model = load_model(model_dir=args.model_dir)

## Get the test data loader
testloader = get_loader(set='test', data_dir=args.data_dir)

test_model(device=device,
           model=trained_validated_model,
           testloader=testloader)
