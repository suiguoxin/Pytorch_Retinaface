import os
import argparse
import json
import torch
from nni.compression.torch import SimulatedAnnealingPruner

from data import cfg_mnet, cfg_re50
from models.retinaface import RetinaFace
from test_widerface import load_model, args


def evaluate(trained_model, network, experiment_data_dir):
    '''
    Parameters:
    -----------
    trained_model : str
        trained state_dict file path to open
    network : str
        Backbone network, mobile0.25 or resnet50
        
    Returns:
    -------
    json
        evaluation result
    '''
    cmd = 'python3 test_widerface.py --trained_model "{}" --network {} \
        && cd ./widerface_evaluate \
        && python3 setup.py build_ext --inplace \
        && python3 evaluation.py --experiment_data_dir {}'.format(trained_model, network, experiment_data_dir)

    os.system(cmd)

    with open(os.path.join(experiment_data_dir, 'evaluation_result.json')) as json_file:
        evaluation_result = json.load(json_file)
        return evaluation_result


if __name__ == '__main__':
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50

    # load pre-trained model
    model = RetinaFace(cfg=cfg, phase = 'test')
    model = load_model(model, args.trained_model, args.cpu)

    def evaluator(model, level='easy'):
        torch.save(model.state_dict(), args.trained_model)
        evaluation_resut = evaluate(trained_model=args.trained_model, network=args.network, experiment_data_dir=args.experiment_data_dir)

        return evaluation_resut[level]
    
    configure_list = [{
        'sparsity': args.sparsity,
        'op_types': ['default']
    }]

    pruner = SimulatedAnnealingPruner(
        model, configure_list, evaluator=evaluator, cool_down_rate=args.cool_down_rate, experiment_data_dir=args.experiment_data_dir)
    pruner.compress()

    pruner.export_model(os.path.join(args.experiment_data_dir, 'model.pth'), os.path.join(args.experiment_data_dir, 'mask.pth')) 
    
    model = RetinaFace(cfg=cfg, phase = 'test')
    model_pruned = load_model(model, os.path.join(args.experiment_data_dir, 'model.pth'), args.cpu)
    
    evaluation_result = evaluator(model_pruned)
    print('Evaluation result : %s' % evaluation_result)
