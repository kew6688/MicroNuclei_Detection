import torch

def stat_score(    preds,
                    target,
                    reduce,
    ):
    """Calculate the number of tp, fp, tn, fn.

    Args:
        preds: An ``(N, C)`` or ``(N, C, X)`` tensor of predictions (0 or 1)
        target: An ``(N, C)`` or ``(N, C, X)`` tensor of true labels (0 or 1)
        reduce: One of ``'micro'``, ``'macro'``, ``'samples'``

    Return:
        Returns a list of 4 tensors; tp, fp, tn, fn.
        The shape of the returned tensors depends on the shape of the inputs
        and the ``reduce`` parameter:

        If inputs are of the shape ``(N, C)``, then:

        - If ``reduce='micro'``, the returned tensors are 1 element tensors
        - If ``reduce='macro'``, the returned tensors are ``(C,)`` tensors
        - If ``reduce='samples'``, the returned tensors are ``(N,)`` tensors

        If inputs are of the shape ``(N, C, X)``, then:

        - If ``reduce='micro'``, the returned tensors are ``(N,)`` tensors
        - If ``reduce='macro'``, the returned tensors are ``(N,C)`` tensors
        - If ``reduce='samples'``, the returned tensors are ``(N,X)`` tensors
    """
    dim: Union[int, List[int]] = 1  # for "samples"
    if reduce == "micro":
        dim = [0, 1] if preds.ndim == 2 else [1, 2]
    elif reduce == "macro":
        dim = 0 if preds.ndim == 2 else 2

    true_pred, false_pred = target == preds, target != preds
    pos_pred, neg_pred = preds == 1, preds == 0

    tp = (true_pred * pos_pred).sum(dim=dim)
    fp = (false_pred * pos_pred).sum(dim=dim)

    tn = (true_pred * neg_pred).sum(dim=dim)
    fn = (false_pred * neg_pred).sum(dim=dim)

    return tp.long(), fp.long(), tn.long(), fn.long()

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    # confusion matrix
    TP, FP, FN, TN = 0,0,0,0

    with torch.no_grad():
        for pack in dataloader:
            X, y = pack['image'].to(device), pack['label'].to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            preds = pred.argmax(1)
            target = y
            true_pred, false_pred = target == preds, target != preds
            pos_pred, neg_pred = preds == 0, preds == 1
            
            dim = 0
            TP += (true_pred * pos_pred).sum(dim=dim).item()
            FP += (false_pred * pos_pred).sum(dim=dim).item()

            TN += (true_pred * neg_pred).sum(dim=dim).item()
            FN += (false_pred * neg_pred).sum(dim=dim).item()
        print(preds.ndim, target.ndim)

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")