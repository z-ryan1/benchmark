import torch
import os
import random
import logging
from contextlib import suppress
from pathlib import Path

# TorchBench imports
from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.util.jit import jit_model
from torchbenchmark.util.torch_feature_checker import check_native_amp
from torchbenchmark.tasks import COMPUTER_VISION

# effdet imports
from effdet import create_model, unwrap_bench, create_loader, create_dataset, create_evaluator
from effdet.data import resolve_input_config, SkipSubset
from effdet.anchors import Anchors, AnchorLabeler

# timm imports
from timm.models.layers import set_layer_config
from timm.optim import create_optimizer
from timm.utils import ModelEmaV2, NativeScaler
from timm.scheduler import create_scheduler

# local imports
from .parser import get_args
from .loader import create_datasets_and_loaders

# setup environment variable
CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(CURRENT_DIR.parent.parent, "data", ".data", "coco2017-minimal")

torch.manual_seed(1337)
random.seed(1337)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

class Model(BenchmarkModel):
    task = COMPUTER_VISION.DETECTION

    # This model doesn't support setting batch size for inference
    def __init__(self, device=None, jit=False, train_bs=1, eval_bs=2):
        super().__init__()
        self.device = device
        self.jit = jit
        # generate arguments
        args = get_args()
        # Use native amp if possible
        args.native_amp = True and check_native_amp()
        # Disable distributed
        args.distributed = False
        args.device = device
        args.torchscript = jit
        args.world_size = 1
        args.rank = 0

        with set_layer_config(scriptable=args.torchscript):
            model = create_model(
                model=args.model,
                bench_task='train',
                num_classes=args.num_classes,
                pretrained=args.pretrained,
                pretrained_backbone=args.pretrained_backbone,
                redundant_bias=args.redundant_bias,
                label_smoothing=args.smoothing,
                legacy_focal=args.legacy_focal,
                jit_loss=args.jit_loss,
                soft_nms=args.soft_nms,
                bench_labeler=args.bench_labeler,
                checkpoint_path=args.initial_checkpoint,
            )
        model_config = model.config  # grab before we obscure with DP/DDP wrappers
        model = model.to(device)
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)

        self.model, self.eval_model = jit_model(model, jit=jit)
        self.optimizer = create_optimizer(args, model)
        self.amp_autocast = suppress
        if args.native_amp:
            self.amp_autocast = torch.cuda.amp.autocast
            loss_scaler = NativeScaler()
        self.model_ema = None
        if args.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            self.model_ema = ModelEmaV2(model, decay=args.model_ema_decay)
        self.lr_scheduler, self.num_epochs = create_scheduler(args, self.optimizer)

        loader_train, loader_eval, evaluator = create_datasets_and_loaders(args, model_config)
        if model_config.num_classes < loader_train.dataset.parser.max_label:
            logging.error(
                f'Model {model_config.num_classes} has fewer classes than dataset {loader_train.dataset.parser.max_label}.')
            exit(1)
        if model_config.num_classes > loader_train.dataset.parser.max_label:
            logging.warning(
                f'Model {model_config.num_classes} has more classes than dataset {loader_train.dataset.parser.max_label}.')


    def get_module(self):
        self.model.eval()

    def train(self, niter=1):
        if not self.device == "cuda":
            raise NotImplementedError("Only CUDA is supported by this model")
        if self.jit:
            raise NotImplementedError("JIT is not supported by this model")

    def eval(self, niter=2):
        if not self.device == "cuda":
            raise NotImplementedError("Only CUDA is supported by this model")
        if self.jit:
            raise NotImplementedError("JIT is not supported by this model")
