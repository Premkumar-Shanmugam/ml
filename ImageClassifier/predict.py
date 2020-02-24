from utils import get_args, get_device, get_class_names
from modeler import load_model, predict

## Get the arguments
args = get_args('predict')

## Get the device
device = get_device(args.gpu)

## Get the model saved as checkpoint
trained_validated_model = load_model(model_dir=args.model_dir)

probs, classes = predict(device=device,
                         image_path=args.image_path,
                         model=trained_validated_model,
                         topx=args.topk)

## Get names of the classes
class_names = get_class_names(classes=classes,
                             cat_names=args.cat_names)

## Print prediction(s)
print(('AI Model\'s top {} prediction(s) are:').format(args.topk))
print('Rank'.ljust(5) + 'Predicted Name'.ljust(25) + 'Probability')
for i, (prob, class_name) in enumerate(zip(probs, class_names)):
    print('{}. {} {}%'.format(str(i+1).rjust(3), class_name.ljust(25), ("%.2f" % round(prob*100, 2)).rjust(6)))
    
