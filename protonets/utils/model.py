from tqdm import tqdm

from protonets.utils import filter_opt
from protonets.models import get_model


def load(opt):
    model_opt = filter_opt(opt, 'model')
    model_name = model_opt['model_name']

    del model_opt['model_name']

    return get_model(model_name, model_opt)


def evaluate(model, data_loader, meters, desc=None):
    model.eval()
    metrics_list = []  # log all loss and acc values

    for field, meter in meters.items():
        meter.reset()

    if desc is not None:
        data_loader = tqdm(data_loader, desc=desc)

    for sample in data_loader:

        _, output = model.loss(sample)
        metrics_list.append({'preds': output['preds'],
                             'true_labels': output['true_labels'],
                             'class': output['class'],
                             'acc_val_ind': output['acc_val_ind']})
        for field, meter in meters.items():
            meter.add(output[field])

    return meters, metrics_list
