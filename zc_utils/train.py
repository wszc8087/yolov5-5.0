import argparse               # 解析命令行参数模块
import logging                # 日志模块
import math                   # 数学公式模块
import os                     # 与操作系统进行交互的模块 包含文件路径操作和解析
import random                 # 生成随机数模块
import sys                    # sys系统模块 包含了与Python解释器和它的环境有关的函数
import time                   # 时间模块 更底层
import warnings               # 发出警告信息模块
from copy import deepcopy     # 深度拷贝模块
from pathlib import Path      # Path将str转换为Path对象 使字符串路径易于操作的模块
from threading import Thread  # 线程操作模块

import numpy as np                # numpy数组操作模块
import torch.distributed as dist  # 分布式训练模块
import torch.nn as nn             # 对torch.nn.functional的类的封装 有很多和torch.nn.functional相同的函数
import torch.nn.functional as F   # PyTorch函数接口 封装了很多卷积、池化等函数
import torch.optim as optim       # PyTorch各种优化算法的库
import torch.optim.lr_scheduler as lr_scheduler  # 学习率模块
import torch.utils.data           # 数据操作模块
import yaml                       # 操作yaml文件模块
from torch.cuda import amp        # PyTorch amp自动混合精度训练模块
from torch.nn.parallel import DistributedDataParallel as DDP  # 多卡训练模块
from torch.utils.tensorboard import SummaryWriter  # tensorboard模块
from tqdm import tqdm  # 进度条模块

FILE = Path(__file__).absolute()  # FILE = WindowsPath 'F:\yolo_v5\yolov5-U\train.py'
# 将'F:/yolo_v5/yolov5-U'加入系统的环境变量  该脚本结束后失效
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution, plot_lr_scheduler, plot_results_overlay
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, de_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from utils.metrics import fitness

# 初始化日志模块
logger = logging.getLogger(__name__)

# pytorch 分布式训练初始化
# https://pytorch.org/docs/stable/elastic/run.html
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # 这个 Worker 是这台机器上的第几个 Worker
RANK = int(os.getenv('RANK', -1))              # 这个 Worker 是全局第几个 Worker
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))   # 总共有几个 Worker

def parse_opt(known=False):
    """
    weights: 权重文件
    cfg: 模型配置文件 包括nc、depth_multiple、width_multiple、anchors、backbone、head等
    data: 数据集配置文件 包括path、train、val、test、nc、names、download等
    hyp: 初始超参文件
    epochs: 训练轮次
    batch-size: 训练批次大小
    img-size: 输入网络的图片分辨率大小
    resume: 断点续训, 从上次打断的训练结果处接着训练  默认False
    nosave: 不保存模型  默认False(保存)      True: only test final epoch
    notest: 是否只测试最后一轮 默认False  True: 只测试最后一轮   False: 每轮训练完都测试mAP
    workers: dataloader中的最大work数（线程个数）
    device: 训练的设备
    single-cls: 数据集是否只有一个类别 默认False

    rect: 训练集是否采用矩形训练  默认False
    noautoanchor: 不自动调整anchor 默认False(自动调整anchor)
    evolve: 是否进行超参进化 默认False
    multi-scale: 是否使用多尺度训练 默认False
    label-smoothing: 标签平滑增强 默认0.0不增强  要增强一般就设为0.1
    adam: 是否使用adam优化器 默认False(使用SGD)
    sync-bn: 是否使用跨卡同步bn操作,再DDP中使用  默认False
    linear-lr: 是否使用linear lr  线性学习率  默认False 使用cosine lr
    cache-image: 是否提前缓存图片到内存cache,以加速训练  默认False
    image-weights: 是否使用图片采用策略(selection img to training by class weights) 默认False 不使用

    bucket: 谷歌云盘bucket 一般用不到
    project: 训练结果保存的根目录 默认是runs/train
    name: 训练结果保存的目录 默认是exp  最终: runs/train/exp
    exist-ok: 如果文件存在就ok不存在就新建或increment name  默认False(默认文件都是不存在的)
    quad: dataloader取数据时, 是否使用collate_fn4代替collate_fn  默认False
    save_period: Log model after every "save_period" epoch    默认-1 不需要log model 信息
    artifact_alias: which version of dataset artifact to be stripped  默认lastest  貌似没用到这个参数？
    local_rank: rank为进程编号  -1且gpu=1时不进行分布式  -1且多块gpu使用DataParallel模式

    entity: wandb entity 默认None
    upload_dataset: 是否上传dataset到wandb tabel(将数据集作为交互式 dsviz表 在浏览器中查看、查询、筛选和分析数据集) 默认False
    bbox_interval: 设置界框图像记录间隔 Set bounding-box image logging interval for W&B 默认-1   opt.epochs // 10
    """
    parser = argparse.ArgumentParser()
    # --------------------------------------------------- 常用参数 ---------------------------------------------
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='True only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='True only test final epoch')
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    # --------------------------------------------------- 数据增强参数 ---------------------------------------------
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', default=False, action='store_true', help='evolve hyperparameters')
    parser.add_argument('--multi-scale', default=True, action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--linear-lr', default=False, action='store_true', help='linear LR')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', default=True, action='store_true', help='use weighted image selection for training')
    # --------------------------------------------------- 其他参数 ---------------------------------------------
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, wins do not modify')
    # --------------------------------------------------- 三个W&B(wandb)参数 ---------------------------------------------
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    # parser.parse_known_args()
    # 作用就是当仅获取到基本设置时，如果运行命令中传入了之后才会获取到的其他配置，不会报错；而是将多出来的部分保存起来，留到后面使用
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def train(hyp, opt, device):
    """
    :params hyp: data/hyps/hyp.scratch.yaml   hyp dictionary
    :params opt: main中opt参数
    :params device: 当前设备
    """
    # ----------------------------------------------- 初始化参数和配置信息 ----------------------------------------------
    # 设置一系列的随机数种子
    init_seeds(1 + RANK)

    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, notest, nosave, workers, = \
        opt.save_dir, opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.notest, opt.nosave, opt.workers

    save_dir = Path(save_dir)  # 保存训练结果的目录  如runs/train/exp18
    wdir = save_dir / 'weights'  # 保存权重路径 如runs/train/exp18/weights
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'  # runs/train/exp18/weights/last.pt
    best = wdir / 'best.pt'  # runs/train/exp18/weights/best.pt
    results_file = save_dir / 'results.txt'  # runs/train/exp18/results.txt

    # Hyperparameters超参
    if isinstance(hyp, str):
        with open(hyp, encoding='utf-8') as f:
            hyp = yaml.safe_load(f)

    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)

    # 保存opt
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Configure
    # 是否需要画图: 所有的labels信息、前三次迭代的barch、训练结果等
    plots = not evolve  # create plots
    cuda = device.type != 'cpu'

    # data_dict: 加载VOC.yaml中的数据配置信息  dict
    with open(data,encoding='utf-8') as f:
        data_dict = yaml.safe_load(f)  # data dict
    # with open(cfg, encoding='utf-8') as f:
    #     # model dict  取到配置文件中每条的信息（没有注释内容）
    #     self.yaml = yaml.safe_load(f)
    # Loggers
    loggers = {'wandb': None, 'tb': None}  # loggers dict
    if RANK in [-1, 0]:
        # TensorBoard
        if not evolve:
            prefix = colorstr('tensorboard: ')  # 彩色打印信息
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            loggers['tb'] = SummaryWriter(str(save_dir))

        # W&B  wandb日志打印相关
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        run_id = run_id if opt.resume else None  # start fresh run if transfer learning
        wandb_logger = WandbLogger(opt, save_dir.stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        if loggers['wandb']:
            data_dict = wandb_logger.data_dict
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # may update weights, epochs if resuming

    nc = 1 if single_cls else int(data_dict['nc'])  # nc: number of classes
    # names: 数据集所有类别的名字
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, data)  # check
    # 当前数据集是否是coco数据集(80个类别)  save_json和coco评价
    is_coco = data.endswith('coco.yaml') and nc == 80  # COCO dataset

    # ============================================== 1、model =================================================
    pretrained = weights.endswith('.pt')
    if pretrained:
        # 使用预训练
        # torch_distributed_zero_first(RANK): 用于同步不同进程对数据读取的上下文管理器
        with torch_distributed_zero_first(RANK):
            # 这里下载是去google云盘下载, 一般会下载失败,所以建议自行去github中下载再放到weights下
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        # 这里加载模型有两种方式，一种是通过opt.cfg 另一种是通过ckpt['model'].yaml
        # 区别在于是否使用resume 如果使用resume会将opt.cfg设为空，按照ckpt['model'].yaml来创建模型
        # 这也影响了下面是否除去anchor的key(也就是不加载anchor), 如果resume则不加载anchor
        # 原因: 保存的模型会保存anchors，有时候用户自定义了anchor之后，再resume，则原来基于coco数据集的anchor会自己覆盖自己设定的anchor
        # 详情参考: https://github.com/ultralytics/yolov5/issues/459
        # 所以下面设置intersect_dicts()就是忽略exclude
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        # 筛选字典中的键值对  把exclude删除
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # 载入模型权重
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        # 不使用预训练
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    # 检查数据集 如果本地没有则从torch库中下载并解压数据集
    with torch_distributed_zero_first(RANK):
        print('检擦数据集')
        check_dataset(data_dict)  # check

    # 数据集路径参数
    train_path, test_path = data_dict['train'], data_dict['val']

    # 冻结权重层，这里只是给了冻结权重层的一个例子, 但是作者并不建议冻结权重层, 训练全部层参数, 可以得到更好的性能, 当然也会更慢
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # ============================================== 2、优化器 =================================================
    # nbs 标称的batch_size,模拟的batch_size 比如默认的话上面设置的opt.batch_size=16 -> nbs=64
    # 也就是模型梯度累计 64/16=4(accumulate) 次之后就更新一次模型 等于变相的扩大了batch_size
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    # 根据accumulate设置超参: 权重衰减参数
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")  # 日志

    # 将模型参数分为三组(weights、biases、bn)来进行分组优化
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    # 选择优化器 并设置pg0(bn参数)的优化方式
    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    # 设置pg1(weights)的优化方式
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    # 设置pg2(biases)的优化方式
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    # 打印log日志 优化信息
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))  # 日志
    # 删除三个变量 优化代码
    del pg0, pg1, pg2

    # ============================================== 3、学习率 =================================================
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        # 使用线性学习率
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        # 使用one cycle 学习率  https://arxiv.org/pdf/1803.09820.pdf
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    # 实例化 scheduler
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    plot_lr_scheduler(optimizer, scheduler, epochs, save_dir=save_dir)  # 画出学习率变化曲线

    # ---------------------------------------------- 训练前最后准备 ------------------------------------------------------
    # EMA
    # 单卡训练: 使用EMA（指数移动平均）对模型的参数做平均, 一种给予近期数据更高权重的平均方法, 以求提高测试指标并增加模型鲁棒。
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # 使用预训练
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # gs: 获取模型最大stride=32   [32 16 8]
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # nl: 有多少个detect 3
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    # 获取训练图片和测试图片分辨率 imgsz=640  imgsz_test=640
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # 是否使用DP mode
    # 如果rank=-1且gpu数量>1则使用DataParallel单机多卡模式  效果并不好（分布不平均）
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # 是否使用DDP mode
    # 如果rank !=-1, 则使用DistributedDataParallel模式  真正的单机单卡（分布平均）
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # SyncBatchNorm  是否使用跨卡BN
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # ============================================== 4、数据加载 ===============================================
    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, stride=gs, single_cls=single_cls,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect,
                                            rank=RANK, workers=workers, image_weights=opt.image_weights,
                                            quad=opt.quad, prefix=colorstr('train: '))



if __name__ == "__main__":
    pass