import torch
from scipy.stats import rankdata

def train_seg_net(seg_model, seg_loader, seg_optimizer, seg_loss_function, device='cpu'):
    """Train segmentation only without the help of selection network

    Args:
        seg_model (nn.Module): Segmentation Network
        seg_loader (nn.Module): Dataloader of the Segmentation Network (drop_last is not mandatory)
        seg_optimizer (_type_): Segmentation Network
        seg_loss_function (torch.optim): loss Function (usually DiceLoss)
        device (str, optional): gpu device. Defaults to 'cpu'.

    Returns:
        float: the average loss of this epoch
    """
    seg_model.to(device)
    seg_model.train()
    step = 0.
    loss_a = 0.
    for img, label in seg_loader:
        # load data to gpu
        img, label = img.to(device), label.to(device)
        # forward pass
        o = seg_model(img)
        # calculate loss
        loss = seg_loss_function(o, label)
        # backward pass
        loss.backward()
        # step and zero_grad
        seg_optimizer.step()
        seg_optimizer.zero_grad()
        # cummulate the loss for monitoring purpose
        step += 1.
        loss_a += loss.item()
    loss_of_this_epoch = loss_a / step
    # return it
    return loss_of_this_epoch


def dice_metric(y_pred, y_true, sigmoid=True, mean=False, eps=1e-6):
    ### Not loss !!!!! Is Metric !!!!!
    '''
    y_pred, y_true -> [N, C=1, H, W]
    perform argmax and convert the predicted segmentation map into on hot format,
    then calculate the dice metric compare with true label
    '''
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = y_pred >= 0.5
    y_pred = y_pred.float()


    numerator = torch.sum(y_true*y_pred, dim=(2, 3)) * 2
    denominator = torch.sum(y_true, dim=(2, 3)) + torch.sum(y_pred, dim=(2, 3)) + eps


    return (numerator / denominator).mean() if mean else numerator / denominator


def evaluate_seg_net(seg_model, test_loader, evaluation_metric=dice_metric, device='cpu'):

    seg_model.to(device)
    seg_model.eval()
    step = 0.
    dice_a = 0.
    with torch.no_grad():
        for img, label in test_loader:
            # load data to gpu
            img, label = img.to(device), label.to(device)
            # forward pass
            o = seg_model(img)
            dice = evaluation_metric(o, label, sigmoid=True, mean=True)
            
            # cummulate the loss for monitoring purpose
            step += 1.
            dice_a += dice.item()
        dice_of_this_epoch = dice_a / step
    # return it
    return dice_of_this_epoch


def weighted_loss(loss_1, loss_2, weight_1, weight_2):
    return weight_1*loss_1 + weight_2*loss_2


def train_seg_net_baseon_sel_net(
        sequence_length,
        num_selection,
        selected_weights,
        other_weights,
        seg_model, 
        sel_model, 
        seg_loader,
        seg_optimizer,
        seg_loss_function,
        device='cpu'
    ):
    # remember the loader should drop the last batch to prevent differenct sequence number in the last batch
    seg_model.train()
    seg_model.to(device)
    sel_model.eval()
    sel_model.to(device)
    step = 0.
    loss_a = 0.
    for img, label in seg_loader:
        img, label = img.to(device), label.to(device)
        # forward pass and calculate the selection
        with torch.no_grad():
            selection_output = sel_model(img)
        # selected image index list
        selection = torch.topk(selection_output.flatten(), k=num_selection).indices.tolist() # returns a list
        # other image index list
        other_index = [i for i in range(sequence_length) if i not in selection]
        selected_img, selected_label = img[selection], label[selection]
        other_img, other_label = img[other_index], label[other_index]

        # forward pass of selected data
        selected_output = seg_model(selected_img)
        selected_loss = seg_loss_function(selected_output, selected_label)
        # forward pass of other data
        other_output = seg_model(other_img)
        other_loss = seg_loss_function(other_output, other_label)

        # combine two losses
        combined_loss = weighted_loss(selected_loss, other_loss, selected_weights, other_weights)
        combined_loss.backward()
        seg_optimizer.step()
        seg_optimizer.zero_grad()

        loss_a += combined_loss.item()
        step += 1.
    loss_of_this_epoch = loss_a / step

    return loss_of_this_epoch
    

def kernel(x, sigma):
    """
    perform element-wise kernel calculate
    :param x: x is already calculated batch-wised
    :param sigma:
    :return:
    """
    return torch.exp(-x**2 / (2*sigma**2))


def term(l_small, l_big, sigma):
    b_small, _ = l_small.shape
    b_big, _ = l_big.shape

    x_0 = l_small.repeat_interleave(b_big, dim=0)
    x_1 = l_big.repeat_interleave(b_small, dim=0)


    # ||x_0 - x_1|| is of shape (b, b, num_dice)
    l2_norm = torch.norm(x_0 - x_1, dim=-1)
    
    k = kernel(l2_norm, sigma)

    
    return k.sum() / (b_small*b_big)


def mmd(distribution_0, distribution_1, sigma):
    """
    distribution's -> [num_sample, num_dice]
    weights' shape -> [num_sample, 1]
    """
    # print(distribution_0.shape, weights_0.shape, distribution_1.shape, weights_1.shape)
    term_1 = term(distribution_0, distribution_0, sigma)
    term_2 = term(distribution_1, distribution_1, sigma)
    term_3 = term(distribution_0, distribution_1, sigma)
    return term_1 + term_2 - 2*term_3


def convert(x):
    x = (x+1.)**3
    return torch.softmax(x, 0)*10



def train_sel_net_baseon_seg_net(sel_model, seg_model, sel_loader, sel_optimizer, device='cpu'):
    # remember the loader should drop the last batch to prevent differenct sequence number in the last batch
    sel_model.train()
    seg_model.eval()
    sel_model.to(device)
    seg_model.to(device)
    step = 0.
    loss_a = 0.
    
    for img, label in sel_loader:
        img, label = img.to(device), label.to(device)
        # forward pass and calculate the performance of each data in the sequence
        with torch.no_grad():
            seg_output = seg_model(img)
            performance = dice_metric(seg_output, label, sigmoid=True, mean=False)
        score = sel_model(img).view(-1, 1)
        
        loss = mmd(score, convert(performance), sigma=3.)
        print(score, convert(performance))
        # backward and zerograd
        loss.backward()

        sel_optimizer.step()
        sel_optimizer.zero_grad()



        

        loss_a += loss.item()
        step += 1.
    loss_of_this_epoch = loss_a / step

    return loss_of_this_epoch




def eval_sel_net_baseon_seg_net(sel_model, seg_model, holdout_loader, device='cpu'):
    # remember the loader should drop the last batch to prevent differenct sequence number in the last batch
    sel_model.eval()
    seg_model.eval()
    sel_model.to(device)
    seg_model.to(device)
    step = 0.
    loss_a = 0.
    
    for img, label in holdout_loader:
        img, label = img.to(device), label.to(device)
        # forward pass and calculate the performance of each data in the sequence
        with torch.no_grad():
            seg_output = seg_model(img)
            performance = dice_metric(seg_output, label, sigmoid=True, mean=False)
        score = sel_model(img).view(-1, 1)
        
        loss = mmd(score, convert(performance), sigma=3.)
        # backward and zerograd
        loss.backward()



        

        loss_a += loss.item()
        step += 1.

    loss_of_this_epoch = loss_a / step

    return loss_of_this_epoch
