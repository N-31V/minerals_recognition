import torch
from core.architecture.experiment.nn_experimenter import FasterRCNNExperimenter
from core.operation.optimization.svd_tools import decompose_module

IMG_SIZE = (1, 3, 922, 1228)


def profile(model, name, input_size=IMG_SIZE):
    model.to('cuda')
    model.eval()
    image = torch.randn(input_size, device='cuda')
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{name}'),
        with_flops=True) as prof:
            outputs = model(image)
    return prof


def flop(profiler) -> int:
    flops = {}
    for e in profiler.events():
        if e.flops != 0:
            if e.key not in flops.keys():
                flops[e.key] = {'flops': e.flops, 'n': 1}
            else:
                flops[e.key]['flops'] += e.flops
                flops[e.key]['n'] +=1
    for k, v in flops.items():
        print(f"{k}: {v['flops']} flops, {v['n']} calls")
    return sum([v['flops'] for v in flops.values()])


def trace(model: torch.nn.Module) -> torch.nn.Module:
    model.to('cuda')
    model.eval()
    return torch.jit.trace(model, torch.randn(*IMG_SIZE, device='cuda'))


def script(model: torch.nn.Module) -> torch.nn.Module:
    model.to('cuda')
    model.eval()
    return torch.jit.script(model)


def convert_ONNX(model: torch.nn.Module, name: str, operator_export_type=torch.onnx.OperatorExportTypes.ONNX):
    model.eval()
    dummy_input = torch.randn(*IMG_SIZE, requires_grad=True, device='cuda')
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=f'{name}.onnx',
        verbose=True,
        operator_export_type=operator_export_type,
        opset_version=12
    )

if __name__== "__main__":
    # baseline = FasterRCNNExperimenter(
    #     num_classes=2,
    #     weights='models/detection/minerals(clean_binary)/FasterRCNN/train.sd.pt'
    # )
    # convert_ONNX(baseline.model, 'FasterRCNN')

    svd = FasterRCNNExperimenter(
        num_classes=2,
    )
    decompose_module(svd.model, decomposing_mode='channel')
    svd.load_model('models/detection/minerals(clean_binary)/FasterRCNN_SVD_channel_O-10_H-0.1/train')
    convert_ONNX(svd.model, 'FasterRCNN_svd')


    # baseline = resnet50(num_classes=7)
    # baseline.load_state_dict(torch.load(
    #     'models/minerals(1920х1080)/ResNet50_lr0.0001/train.sd.pt',
    #      map_location=torch.device('cpu')
    # ))
    # convert_ONNX(baseline, 'baseline')
    #
    # svd = resnet50(num_classes=7)
    # load_svd_state_dict(
    #     model=svd,
    #     decomposing_mode='channel',  # or 'spatial'
    #     state_dict_path='models/minerals(1920х1080)/ResNe50_lrs_SVD_channel_O-100.0_H-0.001000/e_0.9.sd.pt'
    # )
    # convert_ONNX(svd, 'SVD_channel')
    #
    # sfp = load_sfp_resnet_model(
    #     model=resnet50(num_classes=7),
    #     state_dict_path='models/minerals(1920х1080)/ResNe50_lrs_SFP_P-0.20/pruned.sd.pt',
    #     pruning_ratio=0.2,
    #     mk=True,
    # )
    # convert_ONNX(sfp, 'SFP')
