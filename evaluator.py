import os

from nni.compression.torch import SimulatedAnnealingPruner

from models.retinaface import RetinaFace
from test_widerface import load_model


def evaluate(trained_model, network):
    '''
    Parameters:
    -----------
        
    Returns:
    -------
    float
        evaluation result
    '''
    cmd = 'python test_widerface.py --trained_model "{}" --network {} \
        && cd ./widerface_evaluate \
        && python setup.py build_ext --inplace \
        && python evaluation.py'.format(trained_model, network)

    os.system(cmd)

    with open('/mnt/nfs-storage/users/sgx/Retinaface/result/evaluation_result.json') as json_file:
        evaluation_result = json.load(json_file)
        return evaluation_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retinaface')
    parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
    parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--dataset_folder', default='./data/widerface/val/images/', type=str, help='dataset path')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
    parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
    parser.add_argument('--experiment-data-dir', type=str,
                        default='/mnt/nfs-storage/users/sgx/Retinaface/experiment_data/', help='For saving experiment data')
    args = parser.parse_args()

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
        model.load_state_dict(args.trained_model)
        evaluation_resut = evaluate(trained_model=args.trained_model, network=args.network)

        return evaluation_resut[level]
    
    configure_list = [{
        'sparsity': 0.3,
        'op_types': ['default']  # module types to prune
    }]

    pruner = SimulatedAnnealingPruner(
        model, configure_list, evaluator=evaluator, cool_down_rate=0.9, experiment_data_dir=args.experiment_data_dir)
    pruner.compress()

    pruner.export_model('{}model.pth'.format(
        args.experiment_data_dir), '{}mask.pth'.format(args.experiment_data_dir))
    model_pruned = models.mobilenet_v2().to(device)

    model_pruned.load_state_dict(torch.load(
        '{}model.pth'.format(args.experiment_data_dir)))
    evaluation_result = evaluator(model_pruned)
    print('Evaluation result : %s' % evaluation_result)
