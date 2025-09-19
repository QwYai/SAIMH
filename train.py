import argparse
import os
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from copy import deepcopy
from datasets import ImgFile
from model.hashnetwork2 import *
from model.imagerecover import *
from model.textrecover import *
from evaluate import *
from losses import *
from model.ModalityProjection1 import *
from utils import *
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.ticker import MaxNLocator
import scipy.io as sio

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='nuswide')
    parser.add_argument('--batch_size', type=int, default=500, help='number of images in a batch')
    parser.add_argument('--bit', type=int, default=128, help='length of hash codes')
    parser.add_argument('--Epoch_num', type=int, default=200, help='num of Epochs')
    parser.add_argument('--num_epochs', type=int, default=100, help='num of modality recovery Epochs')
    parser.add_argument('--times', type=int, default=1, help='num of times')
    parser.add_argument('--nc', type=int, default=21000, help='complete pairs')
    parser.add_argument('--n1u', type=int, default=0, help='incomplete pairs with only images')
    parser.add_argument('--n2u', type=int, default=0, help='incomplete pairs with only texts')
    parser.add_argument('--lamdaq', type=float, default=0.0, help='intra-modal')
    parser.add_argument('--lamdam', type=float, default=0.5, help='intra-modal')
    
    # 新增的超参数
    return parser.parse_args()

def generate_train_ds(images, texts, labels):
    index = np.arange(labels.shape[0])
    np.random.shuffle(index)
    images = images[index]
    texts = texts[index]
    labels = labels[index]

    datasets = ImgFile(images, texts, labels)
    data_loader = data.DataLoader(dataset=datasets, batch_size=args.batch_size, shuffle=True, num_workers=5)

    image_dim = images.shape[1]
    text_dim = texts.shape[1]
    
    return data_loader, torch.from_numpy(labels).float(), labels.shape[1], image_dim, text_dim, labels.shape[0]


def generate_test_database_ds(images, texts, labels):
    datasets = ImgFile(images, texts, labels)
    data_loader = data.DataLoader(dataset=datasets, batch_size=args.batch_size, shuffle=False, num_workers=5)
    return data_loader, torch.from_numpy(labels).float()
    
def evaluate():
    # Set the model to testing mode
    model.eval()

    database_codes = []
    i = 0
    loss = 0
    for image, text, _ in database_loader:
        image = image.cuda()
        text = text.cuda()
        fusion = model(image, text)
        codes = torch.sign(fusion)
        database_codes.append(codes.data.cpu().numpy())
    database_codes = np.concatenate(database_codes)
    print("data code length:", len(database_codes))
    test_codes = []
    for image, text, _ in test_loader:
        image = image.cuda()
        text = text.cuda()
        fusion = model(image, text)
        codes = torch.sign(fusion)
        test_codes.append(codes.data.cpu().numpy())
    test_codes = np.concatenate(test_codes)

    map_500 = mean_average_precision(torch.from_numpy(test_codes), torch.from_numpy(database_codes), test_labels, database_labels, torch.device('cuda'), 500)
    print(f'mAP@500: {map_500}')
    map_all = mean_average_precision(torch.from_numpy(test_codes), torch.from_numpy(database_codes), test_labels, database_labels, torch.device('cuda'))
    print(f'mAP@All: {map_all}')

    global map_max
    global map_five
    if map_all > map_max:
        map_max = map_all
        # 计算 PR 曲线
        recall_values, precision_values = compute_pr_curve(torch.from_numpy(test_codes), torch.from_numpy(database_codes), test_labels, database_labels, torch.device('cuda'))
        # 绘制 PR 曲线
        plot_pr_curve(recall_values, precision_values)
    if map_500 > map_five:
        map_five = map_500
        sio.savemat('../Consequence/multi-modality_method_comparison/' + str(args.dataset) + '/MMH/' + str(args.bit) + 'bits/hash_codes.mat', {'B_te': test_codes, 'B_db': database_codes, 'L_te': test_labels.numpy(), 'L_db': database_labels.numpy()})
        # sio.savemat('c/' + str(args.c) + '_' + str(t+1) + '.mat', {'map_500': map_500, 'map_all': map_all})
        if args.nc == 2000:
            sio.savemat('./result21/' + str(t+1) + '/bit_' + str(args.bit) + '.mat', {'map_500': map_500, 'map_all': map_all})
        elif args.nc == 4000:
            sio.savemat('./result22/' + str(t+1) + '/bit_' + str(args.bit) + '.mat', {'map_500': map_500, 'map_all': map_all})
        elif args.nc == 6000:
            sio.savemat('./result23/' + str(t+1) + '/bit_' + str(args.bit) + '.mat', {'map_500': map_500, 'map_all': map_all})
        elif args.nc == 8000:
            sio.savemat('./result24/' + str(t+1) + '/bit_' + str(args.bit) + '.mat', {'map_500': map_500, 'map_all': map_all})
        else:
            sio.savemat('./result25/' + str(t+1) + '/bit_' + str(args.bit) + '.mat', {'map_500': map_500, 'map_all': map_all})
    return map_500, map_all


if __name__ == '__main__':
    args = parse_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if args.dataset == 'mirflickr':
        IAll = sio.loadmat('../../Data/ProcessData/MIRFLICKR/image_features_BoW.mat')['image_features'].astype(np.float32)
        TAll = sio.loadmat('../../Data/ProcessData/MIRFLICKR/text_features_BoW.mat')['text_features'].astype(np.float32)
        LAll = sio.loadmat('../../Data/ProcessData/MIRFLICKR/labels_BoW.mat')['labels'].astype(np.float32)
        index = sio.loadmat('../../Data/ProcessData/MIRFLICKR/index_5000.mat')
    elif args.dataset == 'nuswide':
        IAll = sio.loadmat('../../Data/ProcessData/NUSWIDE/image_features_BoW2.mat')['image_features'].astype(np.float32)
        TAll = sio.loadmat('../../Data/ProcessData/NUSWIDE/text_features_BoW2.mat')['text_features'].astype(np.float32)
        LAll = sio.loadmat('../../Data/ProcessData/NUSWIDE/labels_BoW2.mat')['labels'].astype(np.float32)
        index = sio.loadmat('../../Data/ProcessData/NUSWIDE/index_21000.mat')
    elif args.dataset == 'mscoco':
        IAll = sio.loadmat('../../Data/ProcessData/MSCOCO/image_features_clip512.mat')['image_features'].astype(np.float32)
        TAll = sio.loadmat('../../Data/ProcessData/MSCOCO/text_features_clip512.mat')['text_features'].astype(np.float32)
        LAll = sio.loadmat('../../Data/ProcessData/MSCOCO/labels_clip512.mat')['labels'].astype(np.float32)
        index = sio.loadmat('../../Data/ProcessData/MSCOCO/index_18000.mat')
    else:
        print("This dataset does not exist!")

    tau = 0.07
    if LAll.ndim == 1:
        LAll = LAll[:, np.newaxis]  # 转换为二维数组
    # 找到最大标签值
    max_label = int(np.max(LAll))  # 找到最大的标签值
    # 创建one-hot编码
    one_hot_labels = []
    for labels in LAll:
        # 创建一个大小为 (max_label + 1) 的全零数组
        one_hot = np.zeros(max_label, dtype=np.float32)
        # 设置对应的标签索引为1
        for label in labels:
            if(label != 0):
                one_hot[int(label)-1] = 1  # 将标签对应的索引设为1
        
        one_hot_labels.append(one_hot)
    
    # 将结果转换为NumPy数组
    one_hot_labels = np.array(one_hot_labels)
    # LAll = one_hot_labels
    
    indQ = index['indQ'].squeeze()
    indT = index['indT'].squeeze()
    indD = index['indD'].squeeze()
    # the train dataset
    I_tr = IAll[indT]
    T_tr = TAll[indT]
    L_tr = LAll[indT]
    I_te = IAll[indQ]
    T_te = TAll[indQ]
    L_te = LAll[indQ]
    I_db = IAll[indD]
    T_db = TAll[indD]
    L_db = LAll[indD]

    # 模型和优化器的初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化损失和精度记录列表
    train_losses = []
    train_map_max = []
    train_map_five = []

    set_random_seed(42)

    # 训练流程
    
    for t in range(args.times):
        image_projection = AlignProjectionTransformer(4096, 512, 8).to(device)
        text_projection = AlignProjectionTransformer(1000, 512, 5).to(device)
        contrastive_loss_fn = InfoNCE_Loss(temperature=0.07)
        
        # 1. 划分索引（基于 args.nc, args.n1u, args.n2u）
        N_train = I_tr.shape[0]
        idx_comp, idx_imgonly, idx_textonly = create_index_splits(N_train, args.nc, args.n1u, args.n2u)
        
        # 2. 构造三个 dataloader（用于投影训练）
        complete_loader, img_avail_loader, txt_avail_loader, img_idxs, txt_idxs = make_projection_dataloaders(
            I_tr, T_tr, L_tr, idx_comp, idx_imgonly, idx_textonly, batch_size=args.batch_size
        )
        
        # 3. 投影训练（共同优化 cross + intra）
        optimizer_proj = optim.Adam(list(image_projection.parameters()) + list(text_projection.parameters()), lr=1e-4, weight_decay=1e-6)
        image_projection, text_projection = train_projection_networks(
            image_projection, text_projection,
            complete_loader, img_avail_loader, txt_avail_loader,
            contrastive_loss_fn, structure_reg, optimizer_proj,
            device, epochs=25, lambda_intra=args.lamdaq, tau=tau
        )
        
        # 4. 将整个训练集投影（以便训练 predictor）
        # 注意：train_predictors_on_projections 需要 I_tr, T_tr 原始特征与 idx_complete
        img2txt, txt2img = train_predictors_on_projections(image_projection, text_projection, I_tr, T_tr, idx_comp, device,
                                                           hidden_dim=1024, epochs=args.num_epochs, batch_size=256)
        
        # 5. 使用 predictor 在 projection 空间重建缺失模态
        images_proj_all, texts_proj_all, M1, M2 = build_reconstructed_projection_dataset(
            image_projection, text_projection, img2txt, txt2img, I_tr, T_tr, idx_comp, idx_imgonly, idx_textonly, device
        )
        
        # # 6. 用重建后的投影数据构造哈希训练 loader（注意：哈希网络期望输入是 projection-space 特征）
        train_ds_proj = ImgFile(images_proj_all, texts_proj_all, L_tr, M1, M2)  # 若你的 ImgFile 支持传入投影特征
        train_loader_proj = DataLoader(train_ds_proj, batch_size=args.batch_size, shuffle=True, num_workers=5)

        IAll_proj = get_projected_features(image_projection, IAll, batch_size=args.batch_size, device=device)
        TAll_proj = get_projected_features(text_projection, TAll, batch_size=args.batch_size, device=device)

        # del IAll, TAll

        # the query dataset
        I_te = IAll_proj[indQ]
        T_te = TAll_proj[indQ]
        L_te = LAll[indQ]
        # the train dataset
        # I_tr = images_proj_all
        # T_tr = texts_proj_all
        # L_tr = LAll[indT]
        I_tr = IAll_proj[indT]
        T_tr = TAll_proj[indT]
        L_tr = LAll[indT]
        # the retrieval dataset
        I_db = IAll_proj[indD]
        T_db = TAll_proj[indD]
        L_db = LAll[indD]
        del IAll_proj, TAll_proj, LAll

        train_loader, train_labels, label_dim, image_dim, text_dim, num_train = generate_train_ds(I_tr, T_tr, L_tr)
        train_labels = train_labels.cuda()
        
        test_loader, test_labels = generate_test_database_ds(I_te, T_te, L_te)
        database_loader, database_labels = generate_test_database_ds(I_db, T_db, L_db)
        print('Data loader has been generated!Image dimension = %d, text dimension = %d, label dimension = %d.' % (image_dim, text_dim, label_dim))
    
        print('nc = %d' % args.nc)
        print('bit = %d' % args.bit)
    
        map_max = 0
        map_five = 0
        # 模型和优化器的初始化
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # hidden_dim = 4096
        # model = HashingNetwork(image_dim=image_dim, text_dim=text_dim, hidden_dim = hidden_dim, hash_size = args.bit).to(device)

        hidden_dim = [4096,4096]
        model = HashingNetwork(
            image_dim=image_dim,
            text_dim=text_dim,
            hidden_dims=hidden_dim,  # 传入你想要的隐藏层配置
            hash_size=args.bit,
            device=device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
        for epoch in range(args.Epoch_num):
            model.train()
            epoch_loss = 0
            epoch_metric_loss = 0
            epoch_quantization_loss = 0
    
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.Epoch_num}", unit="batch") as pbar:
                for batch_idx, (image_batch, text_batch, labels) in enumerate(pbar):
                    optimizer.zero_grad()
    
                    image_batch = image_batch.to(device)
                    text_batch = text_batch.to(device)
    
                    # 获取哈希码
                    hash_code_continuous = model(image_batch, text_batch)
    
                    # 计算度量损失
                    # metric_loss_val = negative_log_likelihood_similarity_loss(hash_code_continuous, labels)
                    metric_loss_val = contrastive_loss2(hash_code_continuous, labels)
                    # # 计算量化损失
                    quantization_loss_val = quantization_loss1(hash_code_continuous)
                    
                    # 总损失 = 度量损失 + 量化损失
                    total_loss = metric_loss_val + 0.5 * quantization_loss_val
    
                    # 反向传播和优化
                    total_loss.backward()
                    optimizer.step()
    
                    epoch_loss += total_loss.item()
                    epoch_metric_loss += metric_loss_val.item()
                    epoch_quantization_loss += quantization_loss_val.item()
    
                    pbar.set_postfix(total_loss=epoch_loss, metric_loss=epoch_metric_loss, quantization_loss=epoch_quantization_loss)
                    # pbar.set_postfix(total_loss=epoch_loss, metric_loss=epoch_metric_loss)
    
                print(f"Epoch [{epoch+1}/{args.Epoch_num}], "
                      f"Total Loss: {epoch_loss:.4f}, "
                      f"Metric Loss: {epoch_metric_loss:.4f}, "
                      f"Quantization Loss: {epoch_quantization_loss:.4f}")
            # 评估
            map_500, map_all = evaluate()
            print(f"Best Performance 500:{map_five:.4f}")
            print(f"Best Performance all:{map_max:.4f}")

            train_losses.append(epoch_loss)
            train_map_max.append(map_all)
            train_map_five.append(map_500)









