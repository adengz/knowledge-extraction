import torch
from sklearn.metrics import f1_score


@torch.no_grad()
def get_metrics(predicate_logits: torch.Tensor, position_logits: torch.Tensor,
                predicate_hot: torch.Tensor, position_hot: torch.Tensor):
    """
    Calculates F1 scores for a batch.

    Args:
        predicate_logits: batch_size, num_predicates
        position_logits: batch_size, pad_len, num_predicates, 4
        predicate_hot: batch_size, num_predicates
        position_hot: batch_size, pad_len, num_predicates, 4

    Returns:
        Average F1 scores on predicate and position.
    """
    predicate_true = predicate_hot.cpu()
    predicate_pred = torch.sigmoid(predicate_logits).round_().cpu()
    predicate_f1 = f1_score(predicate_true, predicate_pred, average='samples')

    position_true = position_hot.cpu()
    position_pred = torch.sigmoid(position_logits).round_().cpu()
    position_f1s = []
    for i in range(len(predicate_true)):
        predicate_true_i = torch.where(predicate_true[i] == 1)[0]
        position_pred_i = position_pred[i, :, predicate_true_i, :].view(-1)
        position_true_i = position_true[i, :, predicate_true_i, :].view(-1)
        position_f1s.append(f1_score(position_true_i, position_pred_i))

    return predicate_f1, torch.Tensor(position_f1s).mean().item()
