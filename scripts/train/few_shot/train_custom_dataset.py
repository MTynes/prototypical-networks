import os
import json
from functools import partial
from tqdm import tqdm

import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchnet as tnt

from protonets.engine import Engine

import protonets.utils.data as data_utils
import protonets.utils.model as model_utils
import protonets.utils.log as log_utils
import pandas as pd


def main(opt):
    if not os.path.isdir(opt['log.exp_dir']):
        os.makedirs(opt['log.exp_dir'])

    # save opts
    with open(os.path.join(opt['log.exp_dir'], 'opt.json'), 'w') as f:
        json.dump(opt, f)
        f.write('\n')

    trace_file = os.path.join(opt['log.exp_dir'], 'trace.txt')

    # Postprocess arguments
    opt['model.x_dim'] = list(map(int, opt['model.x_dim'].split(',')))
    opt['log.fields'] = opt['log.fields'].split(',')

    torch.manual_seed(1234)
    if opt['data.cuda']:
        torch.cuda.manual_seed(1234)

    if opt['data.trainval']:
        data = data_utils.load(opt, ['trainval'])
        train_loader = data['trainval']
        val_loader = None
    else:
        data = data_utils.load(opt, ['train', 'val'])
        train_loader = data['train']
        val_loader = data['val']

    model = model_utils.load(opt)

    if opt['data.cuda']:
        model.cuda()

    engine = Engine()
    train_metrics = []
    val_metrics = []
    metrics = []
    meters = {'train': {field: tnt.meter.AverageValueMeter() for field in opt['log.fields']}}

    if val_loader is not None:
        meters['val'] = {field: tnt.meter.AverageValueMeter() for field in opt['log.fields']}

    def on_start(state):
        if os.path.isfile(trace_file):
            os.remove(trace_file)
        state['scheduler'] = lr_scheduler.StepLR(state['optimizer'], opt['train.decay_every'], gamma=0.5)

    engine.hooks['on_start'] = on_start

    def on_start_epoch(state):
        for split, split_meters in meters.items():
            for field, meter in split_meters.items():
                meter.reset()
        state['scheduler'].step()

    engine.hooks['on_start_epoch'] = on_start_epoch

    def on_update(state):
        for field, meter in meters['train'].items():
            meter.add(state['output'][field])
        train_metrics.append({'train_loss': state['output']['loss'], 'train_acc': state['output']['acc']})

    engine.hooks['on_update'] = on_update

    def on_end_epoch(hook_state, state):
        if val_loader is not None:
            if 'best_loss' not in hook_state:
                hook_state['best_loss'] = np.inf
            if 'wait' not in hook_state:
                hook_state['wait'] = 0

        if val_loader is not None:
            model_utils.evaluate(state['model'],
                                 val_loader,
                                 meters['val'],
                                 desc="Epoch {:d} valid".format(state['epoch']))

        meter_vals = log_utils.extract_meter_values(meters)
        # metrics.append(meter_vals)
        train_losses = meter_vals['train']['loss']  # .value()[0] #mean
        train_accs = meter_vals['train']['acc']
        val_losses = meter_vals['val']['loss']
        val_accs = meter_vals['val']['acc']
        m = {'train_loss': train_losses, 'train_accuracy': train_accs,
             'val_loss': val_losses, 'val_accs': val_accs}
        metrics.append(m)

        print("Epoch {:02d}: {:s}".format(state['epoch'], log_utils.render_meter_values(meter_vals)))
        meter_vals['epoch'] = state['epoch']
        with open(trace_file, 'a') as f:
            json.dump(meter_vals, f)
            f.write('\n')

        if val_loader is not None:
            if meter_vals['val']['loss'] < hook_state['best_loss']:
                hook_state['best_loss'] = meter_vals['val']['loss']
                print("==> best model (loss = {:0.6f}), saving model...".format(hook_state['best_loss']))

                state['model'].cpu()
                torch.save(state['model'], os.path.join(opt['log.exp_dir'], 'best_model.pt'))
                if opt['data.cuda']:
                    state['model'].cuda()

                hook_state['wait'] = 0
            else:
                hook_state['wait'] += 1

                if hook_state['wait'] > opt['train.patience']:
                    print("==> patience {:d} exceeded".format(opt['train.patience']))
                    state['stop'] = True
        else:
            state['model'].cpu()
            torch.save(state['model'], os.path.join(opt['log.exp_dir'], 'best_model.pt'))
            if opt['data.cuda']:
                state['model'].cuda()

    engine.hooks['on_end_epoch'] = partial(on_end_epoch, {})

    engine.train(
        model=model,
        loader=train_loader,
        optim_method=getattr(optim, opt['train.optim_method']),
        optim_config={'lr': opt['train.learning_rate'],
                      'weight_decay': opt['train.weight_decay']},
        max_epoch=opt['train.epochs']
    )

    model.eval()

    tl = data_utils.load(opt, ['test'])
    test_loader = tl['test']
    if test_loader is not None:
        meters['test'] = {field: tnt.meter.AverageValueMeter() for field in opt['log.fields']}

    test_values, test_preds = model_utils.evaluate(model,
                                                   test_loader,
                                                   meters['test'],
                                                   desc="Evaluating test set....")


    test_loss = test_values['loss'].value()[0]
    test_acc = test_values['acc'].value()[0]
    print('Mean loss: ', '{0:.2f}'.format(test_loss))
    print('Mean accuracy: ', '{0:.2f}%'.format(test_acc * 100))
    classes = [tl['class'] for tl in test_preds]
    cls = []
    for el in classes:
        cls.append([s.split('/')[0] for s in el])

    corrects = [torch.eq(tl['preds'], tl['true_labels']).float() for tl in test_preds]
    corrects_means = [torch.eq(tl['preds'], tl['true_labels']).float().mean() for tl in test_preds]
    #corrects_means_val = [torch.eq(tl['preds'], tl['true_labels']).float().mean().item() for tl in test_preds]

    # get individual corrects as a plain nested list
    corrects_vals = [torch.Tensor.cpu(c).detach().numpy() for c in corrects]

    # Expand classes match to fit with shape of corrects_vals.
    # opt['data.test_query'] default is set to 15
    # Thus there are 15 tests for each sample.
    # Every class in the cls list corresponds to
    # a sample label from the test_preds object

    classes_match = [[[el for i in range(opt['data.test_query'])] for el in group] for group in cls]
    full_class_names_match = [[[el for i in range(opt['data.test_query'])] for el in group] for group in classes]

    pred_df = pd.DataFrame({'class': np.asarray(classes_match).flatten(),
                            'correct': np.asarray(corrects_vals).flatten(),
                            'full_class_name': np.asarray(full_class_names_match).flatten()})
    # use 1 and -1 as class labels so that n*-1 can be used to determine the value
    # for an incorrect prediction using the class label
    pred_df['label'] = pred_df.apply(lambda x: -1 if (x['class'] == 'hc') else 1, axis=1)
    pred_df['prediction'] = pred_df.apply(lambda x: x['label']
    if x['correct'] == 1.0 else x['label'] * -1, axis=1)
    pred_df.replace(-1, 0, inplace=True)

    pred_df.to_csv('predictions_and_classes.csv', index=False)

    df = pd.DataFrame(metrics)
    df.to_csv('model_metrics.csv', index=False)
