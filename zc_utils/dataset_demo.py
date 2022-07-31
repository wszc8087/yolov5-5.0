import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import glob ,os

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

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)
                if p.is_dir():  # dir
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
            img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
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

def img2label_paths(img_paths):
    """用在LoadImagesAndLabels模块的__init__函数中
    根据imgs图片的路径找到对应labels的路径
    Define label paths as a function of image paths
    :params img_paths: {list: 50}  整个数据集的图片相对路径  例如: '..\\datasets\\VOC\\images\\train2007\\000012.jpg'
                                                        =>   '..\\datasets\\VOC\\labels\\train2007\\000012.jpg'
    """
    # 因为python是跨平台的,在Windows上,文件的路径分隔符是'\',在Linux上是'/'
    # 为了让代码在不同的平台上都能运行，那么路径应该写'\'还是'/'呢？ os.sep根据你所处的平台, 自动采用相应的分隔符号
    # sa: '\\images\\'    sb: '\\labels\\'
    s_img, s_label = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    # 把img_paths中所以图片路径中的images替换为labels
    # 使用后跟空格的逗号作为分隔符，将字符串拆分为列表：
    s_label_list =[s_label.join(x.rsplit(s_img,1).rslit('.',1)[0] + '.txt' for x in img_paths)]
    return s_label_list