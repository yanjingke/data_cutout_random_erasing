"""输入为：样本的size和生成的随机lamda值"""


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    """1.论文里的公式2，求出B的rw,rh"""
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    """2.论文里的公式2，求出B的rx,ry（bbox的中心点）"""
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    # 限制坐标区域不超过样本大小

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    """3.返回剪裁B区域的坐标值"""
    return bbx1, bby1, bbx2, bby


for i, (input, target) in enumerate(train_loader):
    # measure data loading time
    data_time.update(time.time() - end)

    input = input.cuda()
    target = target.cuda()
    r = np.random.rand(1)
    if args.beta > 0 and r < args.cutmix_prob:
        # generate mixed sample
        """1.设定lamda的值，服从beta分布"""
        lam = np.random.beta(args.beta, args.beta)
        """2.找到两个随机样本"""
        rand_index = torch.randperm(input.size()[0]).cuda()
        target_a = target  # 一个batch
        target_b = target[rand_index]  # batch中的某一张
        """3.生成剪裁区域B"""
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        """4.将原有的样本A中的B区域，替换成样本B中的B区域"""
        input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        """5.根据剪裁区域坐标框的值调整lam的值"""
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        # compute output
        """6.将生成的新的训练样本丢到模型中进行训练"""
        output = model(input)
        """7.按lamda值分配权重"""
        loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
    else:
        # compute output
        output = model(input)
        loss = criterion(output, target)