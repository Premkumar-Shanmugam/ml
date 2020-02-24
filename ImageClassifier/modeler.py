import torch
from torchvision import models
from torch import nn
from torch import optim

from utils import process_image

def get_model(arch='vgg11',
              l1units=4096,
              l2units=1024):
    ## Use the pre-trained VGG model
    print('Fetching Pre-trained ' + arch + ' model...')
    model = eval('models.' + arch + '(pretrained=True)')
    
    ## Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    ## Define new Classifier
    new_classifier = nn.Sequential(nn.Linear(25088, l1units),
                               nn.ReLU(),
                               nn.Dropout(p=0.3),
                               nn.Linear(l1units, l2units),
                               nn.ReLU(),
                               nn.Dropout(p=0.3),
                               nn.Linear(l2units, 102),
                               nn.LogSoftmax(dim=1))

    ## Fit the classifier to the model
    model.classifier = new_classifier
    print('Model fetched...')
    
    return model

def train_model(model=None,
                device=None,
                epochs=4,
                learn_rate=0.001,
                trainloader=None,
                validloader=None,
                print_every=50):

    if (model == None):
        print('No model to train.')
    elif (trainloader == None):
        print('No data to learn.')
    else:
        print('Training Start...')
        if (validloader == None):
            print('No data to validate learning.')
        model.to(device)
        steps = 0
        running_loss = 0

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
        
        for e in range(epochs):
            for images, labels in trainloader:
                steps += 1
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if (steps % print_every == 0) and (validloader):
                    valid_loss, accuracy, model = validate_model(model=model,
                                                                 device=device,
                                                                 validloader=validloader,
                                                                 criterion=criterion)
                    print(f"Epoch {e+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                          f"Valid accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0
    print('Training End...')
    
    return model

def validate_model(model=None,
                   device='cpu',
                   validloader=None,
                   criterion=None):
    valid_loss = 0
    accuracy = 0
    
    model.eval()
    
    with torch.no_grad():
        for valid_images, valid_labels in validloader:
            valid_images, valid_labels = valid_images.to(device), valid_labels.to(device)
            log_ps_valid = model(valid_images)
            batch_loss = criterion(log_ps_valid, valid_labels)
            valid_loss += batch_loss.item()

            # Accuracy
            ps = torch.exp(log_ps_valid)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == valid_labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    model.train()
    
    return valid_loss, accuracy, model

def save_model(arch=None,
               model=None,
               mapping=None,
               save_dir='./',
               state=''):
    if (model == None):
        print('No model to save.')
        
    ## Define the checkpoint 
    checkpoint = {'arch':       arch,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'mapping':    mapping
                 }
    
    ## Save the checkpoint
    to = save_dir + 'image_classifier_' + state + '_checkpoint.pth'
    torch.save(checkpoint, to)

def load_model(model_dir='./'):
    print('Fetching trained and validated model...')
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
        
    checkpoint = torch.load(model_dir + 'image_classifier_trained_validated_checkpoint.pth', map_location=map_location)
    
    arch = checkpoint ['arch']
    model = eval('models.' + arch + '(pretrained=True)')

    for param in model.parameters(): 
        param.requires_grad = False

    model.classifier = checkpoint ['classifier']
    model.load_state_dict (checkpoint ['state_dict'])
    model.class_to_idx = checkpoint ['mapping']
    print('Model fetched...')
    
    return model

def test_model(device=None,
               model=None,
               testloader=None):
    if (model==None):
        print('No model to test.')
    elif (testloader==None):
        print('No data to test.')
    else:
        print('Testing Start...')
        model.to(device)
        model.eval()
        criterion = nn.NLLLoss()

        test_loss = 0
        test_accuracy = 0

        with torch.no_grad():
            for test_images, test_labels in testloader:
                test_images, test_labels = test_images.to(device), test_labels.to(device)
                log_ps_test = model(test_images)
                batch_loss = criterion(log_ps_test, test_labels)
                test_loss += batch_loss.item()

                # Accuracy
                ps = torch.exp(log_ps_test)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == test_labels.view(*top_class.shape)
                test_accuracy += torch.mean(equals.type(torch.FloatTensor))
        print('Testing End...')
              
        print(f"Test loss: {test_loss/len(testloader):.3f}.. "
              f"Test accuracy: {test_accuracy/len(testloader):.3f}")

def predict(device=None,
            image_path=None,
            model=None,
            topx=1):
    
    ## Process image in the image_path
    np_img = process_image(image_path)
    
    ## Convert from np to tensor
    tensor_img = torch.from_numpy(np_img).type(torch.FloatTensor)
    
    ## Convert tensor to a batch of 1
    tensor_img.unsqueeze_(0)
    
    ## Predict prob for top most prediction
    model.to(device)
    model.eval()
    with torch.no_grad():
        log_ps = model(tensor_img)
    ps = torch.exp(log_ps)
    
    probs, class_idxs = ps.topk(topx, dim=1)
    
    ## Convert tensors to lists
    probs = probs.tolist()[0]
    class_idxs = class_idxs.tolist()[0]
    
    ## Reverse dictionary
    idx_to_class = dict([[v,k] for k,v in model.class_to_idx.items()])
    
    ## Get Classes from Indices
    classes = [idx_to_class[i] for i in class_idxs]
    
    return probs, classes
