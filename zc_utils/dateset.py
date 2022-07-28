import torch
import yaml
from torch.utils.data import Dataset   # 自定义数据集模块
from utils.torch_utils import torch_distributed_zero_first   # 作者的

import random , glob, os ,logging
from pathlib import Path
from tqdm import tqdm

def create_dataloader(path, imgsz, batch_size, stride, single_cls=False,
                      hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      rank=-1, workers=8, image_weights=False, quad=False, prefix=''):
    """在train.py中被调用，用于生成Trainloader, dataset，testloader
    自定义dataloader函数: 调用LoadImagesAndLabels获取数据集(包括数据增强) + 调用分布式采样器DistributedSampler +
                        自定义InfiniteDataLoader 进行永久持续的采样数据
    :param path: 图片数据加载路径 train/test  如: ../datasets/VOC/images/train2007
    :param imgsz: train/test图片尺寸（数据增强后大小） 640
    :param batch_size: batch size 大小 8/16/32
    :param stride: 模型最大stride=32   [32 16 8]
    :param single_cls: 数据集是否是单类别 默认False
    :param hyp: 超参列表dict 网络训练时的一些超参数，包括学习率等，这里主要用到里面一些关于数据增强(旋转、平移等)的系数
    :param augment: 是否要进行数据增强  True
    :param cache: 是否cache_images False
    :param pad: 设置矩形训练的shape时进行的填充 默认0.0
    :param rect: 是否开启矩形train/test  默认训练集关闭 验证集开启
    :param rank:  多卡训练时的进程编号 rank为进程编号  -1且gpu=1时不进行分布式  -1且多块gpu使用DataParallel模式  默认-1
    :param workers: dataloader的numworks 加载数据时的cpu进程数
    :param image_weights: 训练时是否根据图片样本真实框分布权重来选择图片  默认False
    :param quad: dataloader取数据时, 是否使用collate_fn4代替collate_fn  默认False
    :param prefix: 显示信息   一个标志，多为train/val，处理标签时保存cache文件会用到
    """
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    # 主进程实现数据的预读取并缓存，然后其它子进程则从缓存中读取数据并进行一系列运算。
    # 为了完成数据的正常同步, yolov5基于torch.distributed.barrier()函数实现了上下文管理器
    with torch_distributed_zero_first(rank):
        # 载入文件数据(增强数据集)
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix)

    batch_size = min(batch_size, len(dataset))  # bs
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    # 分布式采样器DistributedSampler
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    # 使用InfiniteDataLoader和_RepeatSampler来对DataLoader进行封装, 代替原D先的DataLoader, 能够永久持续的采样数据
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)
    return dataloader, dataset

class LoadImagesAndLabels(Dataset):
    # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False,
                 image_weights=False, cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        """
        初始化过程并没有什么实质性的操作,更多是一个定义参数的过程（self参数）,以便在__getitem()__中进行数据增强操作,所以这部分代码只需要抓住self中的各个变量的含义就算差不多了
        self.img_files: {list: N} 存放着整个数据集图片的相对路径
        self.label_files: {list: N} 存放着整个数据集图片的相对路径
        cache label -> verify_image_label
        self.labels: 如果数据集所有图片中没有一个多边形label  labels存储的label就都是原始label(都是正常的矩形label)
                     否则将所有图片正常gt的label存入labels 不正常gt(存在一个多边形)经过segments2boxes转换为正常的矩形label
        self.shapes: 所有图片的shape
        self.segments: 如果数据集所有图片中没有一个多边形label  self.segments=None
                       否则存储数据集中所有存在多边形gt的图片的所有原始label(肯定有多边形label 也可能有矩形正常label 未知数)
        self.batch: 记载着每张图片属于哪个batch
        self.n: 数据集中所有图片的数量
        self.indices: 记载着所有图片的index
        self.rect=True时self.batch_shapes记载每个batch的shape(同一个batch的图片shape相同)
        """
        # 1、赋值一些基础的self变量 用于后面在__getitem__中调用
        self.img_size = img_size  # 经过数据增强后的数据图片的大小
        self.augment = augment    # 是否启动数据增强 一般训练时打开 验证时关闭
        self.hyp = hyp            # 超参列表
        # 图片按权重采样  True就可以根据类别频率(频率高的权重小,反正大)来进行采样  默认False: 不作类别区分
        self.image_weights = image_weights
        self.rect = False if image_weights else rect  # 是否启动矩形训练 一般训练时关闭 验证时打开 可以加速
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        # mosaic增强的边界值  [-320, -320]
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride      # 最大下采样率 32
        self.path = path          # 图片路径

        # 2、得到path路径下的所有图片的路径self.img_files  这里需要自己debug一下 不会太难
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                # 获取数据集路径path，包含图片路径的txt文件或者包含图片的文件夹路径
                # 使用pathlib.Path生成与操作系统无关的路径，因为不同操作系统路径的‘/’会有所不同
                p = Path(p)  # os-agnostic
                # 如果路径path为包含图片的文件夹路径
                if p.is_dir():  # dir
                    # glob.glab: 返回所有匹配的文件路径列表  递归获取p路径下所有文件
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                # 如果路径path为包含图片路径的txt文件
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()  # 获取图片路径，更换相对路径
                        # 获取数据集路径的上级父目录  os.sep为路径里的分隔符（不同路径的分隔符不同，os.sep可以根据系统自适应）
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            # 破折号替换为os.sep，os.path.splitext(x)将文件名与扩展名分开并返回一个列表
            # 筛选f中所有的图片文件
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {help_url}')

        # 3、根据imgs路径找到labels的路径self.label_files
        self.label_files = img2label_paths(self.img_files)  # labels

        # 4、cache label 下次运行这个脚本的时候直接从cache中取label而不是去文件中取label 速度更快
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')  # cached labels path
        # Check cache
        if cache_path.is_file():
            # 如果有cache文件，直接加载  exists=True: 是否已从cache文件中读出了nf, nm, ne, nc, n等信息
            cache, exists = torch.load(cache_path), True  # load
            # 如果图片版本信息或者文件列表的hash值对不上号 说明本地数据集图片和label可能发生了变化 就重新cache label文件
            if cache.get('version') != 0.3 or cache.get('hash') != get_hash(self.label_files + self.img_files):
                cache, exists = self.cache_labels(cache_path, prefix), False  # re-cache
        else:
            # 否则调用cache_labels缓存标签及标签相关信息
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # 打印cache的结果 nf nm ne nc n = 找到的标签数量，漏掉的标签数量，空的标签数量，损坏的标签数量，总的标签数量
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        # 如果已经从cache文件读出了nf nm ne nc n等信息，直接显示标签信息  msgs信息等
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display all cache results
            if cache['msgs']:
                logging.info('\n'.join(cache['msgs']))  # display all warnings msg
        # 数据集没有标签信息 就发出警告并显示标签label下载地址help_url
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {help_url}'

        # 5、Read cache  从cache中读出最新变量赋给self  方便给forward中使用
        # cache中的键值对最初有: cache[img_file]=[l, shape, segments] cache[hash] cache[results] cache[msg] cache[version]
        # 先从cache中去除cache文件中其他无关键值如:'hash', 'version', 'msgs'等都删除
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        # pop掉results、hash、version、msgs后只剩下cache[img_file]=[l, shape, segments]
        # cache.values(): 取cache中所有值 对应所有l, shape, segments
        # labels: 如果数据集所有图片中没有一个多边形label  labels存储的label就都是原始label(都是正常的矩形label)
        #         否则将所有图片正常gt的label存入labels 不正常gt(存在一个多边形)经过segments2boxes转换为正常的矩形label
        # shapes: 所有图片的shape
        # self.segments: 如果数据集所有图片中没有一个多边形label  self.segments=None
        #                否则存储数据集中所有存在多边形gt的图片的所有原始label(肯定有多边形label 也可能有矩形正常label 未知数)
        # zip 是因为cache中所有labels、shapes、segments信息都是按每张img分开存储的, zip是将所有图片对应的信息叠在一起
        labels, shapes, self.segments = zip(*cache.values())  # segments: 都是[]
        self.labels = list(labels)  # labels to list
        self.shapes = np.array(shapes, dtype=np.float64)  # image shapes to float64
        self.img_files = list(cache.keys())  # 更新所有图片的img_files信息 update img_files from cache result
        self.label_files = img2label_paths(cache.keys())  # 更新所有图片的label_files信息(因为img_files信息可能发生了变化)
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n  # number of images
        self.indices = range(n)  # 所有图片的index

        # 6、为Rectangular Training作准备
        # 这里主要是注意shapes的生成 这一步很重要 因为如果采样矩形训练那么整个batch的形状要一样 就要计算这个符合整个batch的shape
        # 而且还要对数据集按照高宽比进行排序 这样才能保证同一个batch的图片的形状差不多相同 再选则一个共同的shape代价也比较小
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()  # 根据高宽比排序
            self.img_files = [self.img_files[i] for i in irect]      # 获取排序后的img_files
            self.label_files = [self.label_files[i] for i in irect]  # 获取排序后的label_files
            self.labels = [self.labels[i] for i in irect]            # 获取排序后的labels
            self.shapes = s[irect]                                   # 获取排序后的wh
            ar = ar[irect]                                           # 获取排序后的aspect ratio

            # 计算每个batch采用的统一尺度 Set training image shapes
            shapes = [[1, 1]] * nb    # nb: number of batches
            for i in range(nb):
                ari = ar[bi == i]     # bi: batch index
                mini, maxi = ari.min(), ari.max()   # 获取第i个batch中，最小和最大高宽比
                # 如果高/宽小于1(w > h)，将w设为img_size（保证原图像尺度不变进行缩放）
                if maxi < 1:
                    shapes[i] = [maxi, 1]   # maxi: h相对指定尺度的比例  1: w相对指定尺度的比例
                # 如果高/宽大于1(w < h)，将h设置为img_size（保证原图像尺度不变进行缩放）
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            # 计算每个batch输入网络的shape值(向上设置为32的整数倍)
            # 要求每个batch_shapes的高宽都是32的整数倍，所以要先除以32，取整再乘以32（不过img_size如果是32倍数这里就没必要了）
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # 7、是否需要cache image 一般是False 因为RAM会不足  cache label还可以 但是cache image就太大了 所以一般不用
        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(num_threads).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # img, hw_original, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        """用在__init__函数中  cache数据集label
        加载label信息生成cache文件   Cache dataset labels, check images and read shapes
        :params path: cache文件保存地址
        :params prefix: 日志头部信息(彩打高亮部分)
        :return x: cache中保存的字典
               包括的信息有: x[im_file] = [l, shape, segments]
                          一张图片一个label相对应的保存到x, 最终x会保存所有图片的相对路径、gt框的信息、形状shape、所有的多边形gt信息
                              im_file: 当前这张图片的path相对路径
                              l: 当前这张图片的所有gt框的label信息(不包含segment多边形标签) [gt_num, cls+xywh(normalized)]
                              shape: 当前这张图片的形状 shape
                              segments: 当前这张图片所有gt的label信息(包含segment多边形标签) [gt_num, xy1...]
                           hash: 当前图片和label文件的hash值  1
                           results: 找到的label个数nf, 丢失label个数nm, 空label个数ne, 破损label个数nc, 总img/label个数len(self.img_files)
                           msgs: 所有数据集的msgs信息
                           version: 当前cache version
        """
        x = {}  # 初始化最终cache中保存的字典dict
        # 初始化number missing, found, empty, corrupt, messages
        # 初始化整个数据集: 漏掉的标签(label)总数量, 找到的标签(label)总数量, 空的标签(label)总数量, 错误标签(label)总数量, 所有错误信息
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."  # 日志
        # 多进程调用verify_image_label函数
        with Pool(num_threads) as pool:
            # 定义pbar进度条
            # pool.imap_unordered: 对大量数据遍历多进程计算 返回一个迭代器
            # 把self.img_files, self.label_files, repeat(prefix) list中的值作为参数依次送入(一次送一个)verify_image_label函数
            pbar = tqdm(pool.imap_unordered(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix))),
                        desc=desc, total=len(self.img_files))
            # im_file: 当前这张图片的path相对路径
            # l: [gt_num, cls+xywh(normalized)]
            #    如果这张图片没有一个segment多边形标签 l就存储原label(全部是正常矩形标签)
            #    如果这张图片有一个segment多边形标签  l就存储经过segments2boxes处理好的标签(正常矩形标签不处理 多边形标签转化为矩形标签)
            # shape: 当前这张图片的形状 shape
            # segments: 如果这张图片没有一个segment多边形标签 存储None
            #           如果这张图片有一个segment多边形标签 就把这张图片的所有label存储到segments中(若干个正常gt 若干个多边形标签) [gt_num, xy1...]
            # nm_f(nm): number missing 当前这张图片的label是否丢失         丢失=1    存在=0
            # nf_f(nf): number found 当前这张图片的label是否存在           存在=1    丢失=0
            # ne_f(ne): number empty 当前这张图片的label是否是空的         空的=1    没空=0
            # nc_f(nc): number corrupt 当前这张图片的label文件是否是破损的  破损的=1  没破损=0
            # msg: 返回的msg信息  label文件完好=‘’  label文件破损=warning信息
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f  # 累加总number missing label
                nf += nf_f  # 累加总number found label
                ne += ne_f  # 累加总number empty label
                nc += nc_f  # 累加总number corrupt label
                if im_file:
                    x[im_file] = [l, shape, segments]  # 信息存入字典 key=im_file  value=[l, shape, segments]
                if msg:
                    msgs.append(msg)  # 将msg加入总msg
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"  # 日志
        pbar.close()  # 关闭进度条
        # 日志打印所有msg信息
        if msgs:
            logging.info('\n'.join(msgs))
        # 一张label都没找到 日志打印help_url下载地址
        if nf == 0:
            logging.info(f'{prefix}WARNING: No labels found in {path}. See {help_url}')
        x['hash'] = get_hash(self.label_files + self.img_files)  # 将当前图片和label文件的hash值存入最终字典dist
        x['results'] = nf, nm, ne, nc, len(self.img_files)  # 将nf, nm, ne, nc, len(self.img_files)存入最终字典dist
        x['msgs'] = msgs  # 将所有数据集的msgs信息存入最终字典dist
        x['version'] = 0.3  # 将当前cache version存入最终字典dist
        try:
            torch.save(x, path)  # save cache to path
            logging.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            logging.info(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # path not writeable
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        """
        这部分是数据增强函数，一般一次性执行batch_size次。
        训练 数据增强: mosaic(random_perspective) + hsv + 上下左右翻转
        测试 数据增强: letterbox
        :return torch.from_numpy(img): 这个index的图片数据(增强后) [3, 640, 640]
        :return labels_out: 这个index图片的gt label [6, 6] = [gt_num, 0+class+xywh(normalized)]
        :return self.img_files[index]: 这个index图片的路径地址
        :return shapes: 这个batch的图片的shapes 测试时(矩形训练)才有  验证时为None   for COCO mAP rescaling
        """
        # 这里可以通过三种形式获取要进行数据增强的图片index  linear, shuffled, or image_weights
        index = self.indices[index]

        hyp = self.hyp  # 超参 包含众多数据增强超参
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        # mosaic增强 对图像进行4张图拼接训练  一般训练时运行
        # mosaic + MixUp
        if mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            # img, labels = load_mosaic9(self, index)
            shapes = None

            # MixUp augmentation
            # mixup数据增强
            if random.random() < hyp['mixup']:  # hyp['mixup']=0 默认为0则关闭 默认为1则100%打开
                # *load_mosaic(self, random.randint(0, self.n - 1)) 随机从数据集中任选一张图片和本张图片进行mixup数据增强
                # img:   两张图片融合之后的图片 numpy (640, 640, 3)
                # labels: 两张图片融合之后的标签label [M+N, cls+x1y1x2y2]
                img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1)))

                # 测试代码 测试MixUp效果
                # cv2.imshow("MixUp", img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # print(img.shape)   # (640, 640, 3)

        # 否则: 载入图片 + Letterbox  (val)
        else:
            # Load image
            # 载入图片  载入图片后还会进行一次resize  将当前图片的最长边缩放到指定的大小(512), 较小边同比例缩放
            # load image img=(343, 512, 3)=(h, w, c)  (h0, w0)=(335, 500)  numpy  index=4
            # img: resize后的图片   (h0, w0): 原始图片的hw  (h, w): resize后的图片的hw
            # 这一步是将(335, 500, 3) resize-> (343, 512, 3)
            img, (h0, w0), (h, w) = load_image(self, index)

            # 测试代码 测试load_image效果
            # cv2.imshow("load_image", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print(img.shape)   # (640, 640, 3)

            # Letterbox
            # letterbox之前确定这张当前图片letterbox之后的shape  如果不用self.rect矩形训练shape就是self.img_size
            # 如果使用self.rect矩形训练shape就是当前batch的shape 因为矩形训练的话我们整个batch的shape必须统一(在__init__函数第6节内容)
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            # letterbox 这一步将第一步缩放得到的图片再缩放到当前batch所需要的尺度 (343, 512, 3) pad-> (384, 512, 3)
            # (矩形推理需要一个batch的所有图片的shape必须相同，而这个shape在init函数中保持在self.batch_shapes中)
            # 这里没有缩放操作，所以这里的ratio永远都是(1.0, 1.0)  pad=(0.0, 20.5)
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # 图片letterbox之后label的坐标也要相应变化  根据pad调整label坐标 并将归一化的xywh -> 未归一化的xyxy
            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            # 测试代码 测试letterbox效果
            # cv2.imshow("letterbox", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print(img.shape)   # (640, 640, 3)

        if self.augment:
            # Augment imagespace
            if not mosaic:
                # 不做mosaic的话就要做random_perspective增强 因为mosaic函数内部执行了random_perspective增强
                # random_perspective增强: 随机对图片进行旋转，平移，缩放，裁剪，透视变换
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

            # 色域空间增强Augment colorspace
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # 测试代码 测试augment_hsv效果
            # cv2.imshow("augment_hsv", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print(img.shape)   # (640, 640, 3)

            # Apply cutouts 随机进行cutout增强 0.5的几率使用  这里可以自行测试
            if random.random() < hyp['cutout']:  # hyp['cutout']=0  默认为0则关闭 默认为1则100%打开
                labels = cutout(img, labels)

                # 测试代码 测试cutout效果
                # cv2.imshow("cutout", img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # print(img.shape)   # (640, 640, 3)

        nL = len(labels)  # number of labels
        if nL:
            # xyxy to xywh normalized
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0])

        # 平移增强 随机左右翻转 + 随机上下翻转
        if self.augment:
            # 随机上下翻转 flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)  # np.flipud 将数组在上下方向翻转。
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]   # 1 - y_center  label也要映射

            # 随机左右翻转 flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)   # np.fliplr 将数组在左右方向翻转
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]   # 1 - x_center  label也要映射

        # 6个值的tensor 初始化标签框对应的图片序号, 配合下面的collate_fn使用
        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)  # numpy to tensor

        # Convert BGR->RGB  HWC->CHW
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3 x img_height x img_width
        img = np.ascontiguousarray(img)  # img变成内存连续的数据  加快运算

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        """这个函数会在create_dataloader中生成dataloader时调用：
        整理函数  将image和label整合到一起
        :return torch.stack(img, 0): 如[16, 3, 640, 640] 整个batch的图片
        :return torch.cat(label, 0): 如[15, 6] [num_target, img_index+class_index+xywh(normalized)] 整个batch的label
        :return path: 整个batch所有图片的路径
        :return shapes: (h0, w0), ((h / h0, w / w0), pad)    for COCO mAP rescaling
        pytorch的DataLoader打包一个batch的数据集时要经过此函数进行打包 通过重写此函数实现标签与图片对应的划分，一个batch中哪些标签属于哪一张图片,形如
            [[0, 6, 0.5, 0.5, 0.26, 0.35],
             [0, 6, 0.5, 0.5, 0.26, 0.35],
             [1, 6, 0.5, 0.5, 0.26, 0.35],
             [2, 6, 0.5, 0.5, 0.26, 0.35],]
           前两行标签属于第一张图片, 第三行属于第二张。。。
        """
        # img: 一个tuple 由batch_size个tensor组成 整个batch中每个tensor表示一张图片
        # label: 一个tuple 由batch_size个tensor组成 每个tensor存放一张图片的所有的target信息
        #        label[6, object_num] 6中的第一个数代表一个batch中的第几张图
        # path: 一个tuple 由4个str组成, 每个str对应一张图片的地址信息
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        # 返回的img=[batch_size, 3, 736, 736]
        #      torch.stack(img, 0): 将batch_size个[3, 736, 736]的矩阵拼成一个[batch_size, 3, 736, 736]
        # label=[target_sums, 6]  6：表示当前target属于哪一张图+class+x+y+w+h
        #      torch.cat(label, 0): 将[n1,6]、[n2,6]、[n3,6]...拼接成[n1+n2+n3+..., 6]
        # 这里之所以拼接的方式不同是因为img拼接的时候它的每个部分的形状是相同的，都是[3, 736, 736]
        # 而我label的每个部分的形状是不一定相同的，每张图的目标个数是不一定相同的（label肯定也希望用stack,更方便,但是不能那样拼）
        # 如果每张图的目标个数是相同的，那我们就可能不需要重写collate_fn函数了
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        """同样在create_dataloader中生成dataloader时调用：
        这里是yolo-v5作者实验性的一个代码 quad-collate function 当train.py的opt参数quad=True 则调用collate_fn4代替collate_fn
        作用:  如之前用collate_fn可以返回图片[16, 3, 640, 640] 经过collate_fn4则返回图片[4, 3, 1280, 1280]
              将4张mosaic图片[1, 3, 640, 640]合成一张大的mosaic图片[1, 3, 1280, 1280]
              将一个batch的图片每四张处理, 0.5的概率将四张图片拼接到一张大图上训练, 0.5概率直接将某张图片上采样两倍训练
        """
        # img: 整个batch的图片 [16, 3, 640, 640]
        # label: 整个batch的label标签 [num_target, img_index+class_index+xywh(normalized)]
        # path: 整个batch所有图片的路径
        # shapes: (h0, w0), ((h / h0, w / w0), pad)    for COCO mAP rescaling
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4  # collate_fn4处理后这个batch中图片的个数
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]  # 初始化

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4  # 采样 [0, 4, 8, 16]
            if random.random() < 0.5:
                # 随机数小于0.5就直接将某张图片上采样两倍训练
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                # 随机数大于0.5就将四张图片(mosaic后的)拼接到一张大图上训练
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        # 后面返回的部分和collate_fn就差不多了 原因和解释都写在上一个函数了 自己debug看一下吧
        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4



def check_dataset(data, autodownload=True):
    """用在train.py和detect.py中 检查本地有没有数据集
    检查数据集 如果本地没有则从torch库中下载并解压数据集
    :params data: 是一个解析过的data_dict   len=7
                  例如: ['path'='../datasets/coco128', 'train','val', 'test', 'nc', 'names', 'download']
    :params autodownload: 如果本地没有数据集是否需要直接从torch库中下载数据集  默认True
    """
    # path: WindowPath '..\datasets\coco128'
    path = Path(data.get('path', ''))  # optional 'path' field
    # 如果path不为空 就更新(扩展)train、val和test的路径
    # train: data['train'] -> path / data['train']
    #        'images/train2017' -> '..\\datasets\\coco128\\images\\train2017'
    # val: data['val'] -> path / data['val']
    #        'images/train2017' -> '..\\datasets\\coco128\\images\\train2017'
    if path:
        for k in 'train', 'val', 'test':  #
            if data.get(k):  # prepend path
                data[k] = str(path / data[k]) if isinstance(data[k], str) else [str(path / x) for x in data[k]]

    # train: 训练路径  '..\\datasets\\coco128\\images\\train2017'
    # val: 验证路径    '..\\datasets\\coco128\\images\\train2017'
    # test: 测试路径   None
    # s: 下载地址      'https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip'
    train, val, test, s = [data.get(x) for x in ('train', 'val', 'test', 'download')]
    if val:
        # path.resolve() 该方法将一些的 路径/路径段 解析为绝对路径
        # val: [WindowsPath('E:/yolo_v5/datasets/coco128/images/train2017')]
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        # 如果val不存在 说明本地不存在数据集
        if not all(x.exists() for x in val):
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
            # 如果下载地址s和下载标记(flag)autodownload不为空, 就直接下载
            if s and autodownload:  # download script
                # 如果下载地址s是http开头就从url中下载数据集
                if s.startswith('http') and s.endswith('.zip'):
                    # f: 得到下载文件的文件名 filename
                    f = Path(s).name
                    print(f'Downloading {s} ...')
                    # 开始下载 利用torch.hub.download_url_to_file函数从s路径中下载文件名为f的文件
                    torch.hub.download_url_to_file(s, f)
                    root = path.parent if 'path' in data else '..'  # unzip directory i.e. '../'
                    Path(root).mkdir(parents=True, exist_ok=True)  # create root
                    # 执行解压命名 将文件f解压到root地址 解压后文件名为f
                    r = os.system(f'unzip -q {f} -d {root} && rm {f}')  # unzip
                # 如果下载地址s是bash开头就使用bash指令下载数据集
                elif s.startswith('bash '):  # bash script
                    print(f'Running {s} ...')
                    # 使用bash命令下载
                    r = os.system(s)
                # 否则下载地址就是一个python脚本 执行python脚本下载数据集
                else:  # python script
                    r = exec(s, {'yaml': data})  # return None
                print('Dataset autodownload %s\n' % ('success' if r in (0, None) else 'failure'))  # print result
            else:
                # 下载地址为空 或者不需要下载 标记(flag)autodownload
                raise Exception('Dataset not found.')

def zc_test():
    data = '../data/coco128.yaml'# opt.data
    # data_dict: 加载VOC.yaml中的数据配置信息  dict
    with open(data,encoding='utf-8') as f:
        data_dict = yaml.safe_load(f)  # data dict

    # 检查数据集 如果本地没有则从torch库中下载并解压数据集
    # with torch_distributed_zero_first(RANK):
    check_dataset(data_dict)  # check
    train_path = data_dict['train']
    imgsz= 640

    dataloader, dataset = create_dataloader(train_path,
                                            imgsz,
                                            batch_size // WORLD_SIZE,
                                            gs,
                                            single_cls,
                                            hyp=hyp,
                                            augment=True,
                                            cache=opt.cache_images,
                                            rect=opt.rect,
                                            rank=RANK,
                                            workers=workers,
                                            image_weights=opt.image_weights,
                                            quad=opt.quad,
                                            prefix=colorstr('train: ')
                                            )
if __name__== "__main__":
    zc_test()