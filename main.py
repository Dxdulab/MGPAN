import os
import logging
import torch
import numpy as np
from model import MGPAN 
import dgl
from dgl.data import DGLDataset
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils.data_loader import GraphDataset, GraphmpDataset,check_tensor,extract_topk_attention
from utils.data_loader import augment_dataset, build_metapath_graphs, build_id_maps_and_features_with_rank_optimized, build_node_id_global, set_seed
from utils.metrics import plot_training_curves, plot_test_evaluation, plot_testcv_evaluation
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd  
import pickle
import copy
from config import DEFAULT_METAPATHS, RELATIONS, parse_args

def main(args):
    seed = args.seed
    set_seed(seed)

    device_name = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    device = torch.device(device_name)
    dataset = args.dataset
    graphdata = args.graphdata
    metadata = args.metadata
    metapath = args.metapath
    out_model_dir = os.path.join(args.saved_model_dir, dataset, 'metapaths', args.log)
    os.makedirs(out_model_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, dataset, 'metapaths')
    os.makedirs(log_path, exist_ok=True)
    
    log_fname = os.path.join(log_path, f'{args.log}.out')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        filename=log_fname,
        filemode='a'
    )
    logging.info(args)

    all_graphs, data_labels_dict = dgl.load_graphs(os.path.join(args.data_dir, dataset, graphdata))
    all_labels = data_labels_dict['labels']
    
    with open(os.path.join(args.data_dir, dataset, metadata), 'rb') as f:
        all_meta = pickle.load(f)
    all_graphs_homo = []
    for idx,g in enumerate(all_graphs):
        g_homo = dgl.to_homogeneous(
            g,
            ndata=['feat'],    # 保留节点特征
            edata=['weight'],  # 保留边权重
            store_type=True    # 保留原始节点类型信息
        )
        check_tensor(g_homo.ndata['feat'], f"raw feat graph {idx}")
        all_graphs_homo.append(g_homo)

    all_feats = build_id_maps_and_features_with_rank_optimized(all_graphs_homo)

        
    for idx, (g, f) in enumerate(zip(all_graphs_homo, all_feats)):
        check_tensor(f, f"processed feat graph {idx}")
        g.ndata['feat'] = f

    subject_ids = pd.read_csv(os.path.join(args.data_dir, dataset, args.subject_ids))['subject_id'].tolist()


    assert len(subject_ids) == len(all_graphs), \
        f"Subject 数量 {len(subject_ids)} != 图数量 {len(all_graphs)}，请检查顺序是否对齐"


    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=args.fold_shuffle, random_state=seed)
    feat_dim = args.feat_dim  # 每种节点初始特征维度

    fold_results = []
    all_fpr, all_tpr, all_prec, all_recall,all_aupr = [], [], [], [],[]
    all_auc, all_pr_auc, all_cm = [], [], []

    all_folds = list(sgkf.split(
        X=np.zeros(len(all_labels)),
        y=all_labels.numpy(),
        groups=subject_ids
    ))

    n_folds = len(all_folds)
    all_attention_results = []
    all_raw_predictions_dfs = []
    split_point = max(0, n_folds - args.run_last_folds_first)
    run_order = list(range(split_point, n_folds)) + list(range(0, split_point))
    for fold in run_order:
        train_idx, test_idx = all_folds[fold]
        logging.info(args)

        train_graphs_fold = [copy.deepcopy(all_graphs_homo[i]) for i in train_idx]
        test_graphs_fold  = [copy.deepcopy(all_graphs_homo[i]) for i in test_idx]

        train_labels_fold = [all_labels[i] for i in train_idx]
        test_labels_fold  = [all_labels[i] for i in test_idx]
        
        train_labels_tensor = torch.stack(train_labels_fold)
        test_labels_tensor = torch.stack(test_labels_fold)

        train_meta_fold = [all_meta[i] for i in train_idx]
        test_meta_fold  = [all_meta[i] for i in test_idx]

        unique_train, counts_train = np.unique(train_labels_fold, return_counts=True)
        unique_test, counts_test = np.unique(test_labels_fold, return_counts=True)

        logging.info(f"训练集样本数: {len(train_labels_fold)}，测试集样本数: {len(test_labels_fold)}")

        logging.info("训练集类别分布:")
        for u, c in zip(unique_train, counts_train):
            logging.info(f"  标签 {u}: {c} ({c / len(train_labels_fold):.2%})")

        logging.info("测试集类别分布:")
        for u, c in zip(unique_test, counts_test):
            logging.info(f"  标签 {u}: {c} ({c / len(test_labels_fold):.2%})")
        
        node2id_global, train_graphs_fold1, test_graphs_fold1 = \
            build_node_id_global(
                train_graphs_fold, test_graphs_fold,
                train_meta_fold, test_meta_fold,
                device=args.graph_build_device
            )
        for idx, g in enumerate(train_graphs_fold1):
            check_tensor(g.ndata['feat'], f"train graph {idx} feat")
        for idx, g in enumerate(test_graphs_fold1):
            check_tensor(g.ndata['feat'], f"test graph {idx} feat")
        num_node_ids=len(node2id_global)

        train_dataset_fold = GraphDataset('my_dataset', train_graphs_fold1, train_labels_tensor)
        test_dataset_fold = GraphDataset('my_dataset', test_graphs_fold1, test_labels_tensor)

        train_dataset_aug = augment_dataset(train_dataset_fold,
                                            edge_drop_prob=args.edge_drop_prob,
                                            node_drop_prob=args.node_drop_prob,
                                            feat_mask_prob=args.feat_mask_prob)
    

        metapaths = [path[:] for path in DEFAULT_METAPATHS]

        metapath_cache_dir = os.path.join(args.metapath_dir, dataset, metapath)
        train_save_path = os.path.join(metapath_cache_dir, f'fold_{fold}_train.pkl')
        test_save_path = os.path.join(metapath_cache_dir, f'fold_{fold}_test.pkl')

        if os.path.exists(train_save_path) and os.path.exists(test_save_path):
            print(f"✅ 已找到缓存，加载 Fold {fold+1} 的 metapath 图...")
            with open(train_save_path, 'rb') as f:
                train_dataset_aug, train_mp_graphs= pickle.load(f)
            with open(test_save_path, 'rb') as f:
                test_dataset_fold, test_mp_graphs = pickle.load(f)
        else:
            print(f"⚙️ 未找到缓存，正在生成 Fold {fold+1} 的 metapath 图...")
            
            train_mp_graphs, test_mp_graphs = [], []
        
            for g in train_dataset_aug.graphs:
                if not hasattr(g, 'mp_graphs'):
                    mp_list = build_metapath_graphs(g, metapaths)
                    train_mp_graphs.append(mp_list)

            for g in test_dataset_fold.graphs:  # 测试集保持原图
                if not hasattr(g, 'mp_graphs'):
                    mp_list = build_metapath_graphs(g, metapaths, device=args.graph_build_device)
                    test_mp_graphs.append(mp_list)

            os.makedirs(metapath_cache_dir, exist_ok=True)
            with open(train_save_path, 'wb') as f:
                pickle.dump((train_dataset_aug, train_mp_graphs), f)
            with open(test_save_path, 'wb') as f:
                pickle.dump((test_dataset_fold, test_mp_graphs), f)
            print(f"💾 Fold {fold+1} 的 metapath 图已保存")
        
        train_datasetmp_fold = GraphmpDataset(
            name='train_dataset',
            graphs=train_dataset_aug.graphs,
            labels=train_dataset_aug.labels,
            mp_graphs_list=train_mp_graphs,
        )

        test_datasetmp_fold = GraphmpDataset(
            name='test_dataset',
            graphs=test_dataset_fold.graphs,
            labels=test_dataset_fold.labels,
            mp_graphs_list=test_mp_graphs,
        )


        relations = list(RELATIONS)
        feat_dim = args.feat_dim
        model = MGPAN(
            gnn_type=args.gnn,
            num_gnn_layers=args.num_gnn_layer,
            relations=relations,
            feat_dim=feat_dim,
            embed_dim=args.embed_dim,
            dim_a=args.dim_a,
            dropout1=args.dropout1,
            dropout2=args.dropout2,
            dropout3=args.dropout3,
            attdropout=args.attdropout,
            activation=args.activation,
            num_node_types=args.num_node_types,
            type_emb_dim=args.type_emb_dim,
            num_node_ids=num_node_ids,
            node_id_emb_dim=args.node_id_emb_dim,
            abundance_proj_dim=args.abundance_proj_dim,
            metapaths=metapaths,
            type_emb_hidden_dim=args.type_emb_hidden_dim,
            abundance_input_dim=args.abundance_input_dim,
            classifier_dropout=args.classifier_dropout,
            graph_pool_hidden_dim=args.graph_pool_hidden_dim,
            graph_readout_num_types=args.graph_readout_num_types,
            gat_num_heads=args.gat_num_heads,
            sage_aggregator=args.sage_aggregator,
            residual_dropout=args.residual_dropout
        )

        num_pos = torch.sum(train_labels_tensor  == 1).item()
        num_neg = torch.sum(train_labels_tensor == 0).item()
        if args.auto_pos_class_weight:
            pos_class_weight = num_neg / max(num_pos, 1)
        else:
            pos_class_weight = args.pos_class_weight
        logging.info(f"pos_class_weight: {pos_class_weight}")
        train_acc, train_p, train_r, train_f1, train_auc, train_loss,train_aupr = model.train_model(
            train_datasetmp_fold,
            test_datasetmp_fold,
            fold=fold,
            batch_size=args.batch_size,
            EPOCHS=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            accum_steps=args.accum_steps,
            num_workers=args.num_workers,
            pos_class_weight=pos_class_weight,
            device=device,
            model_dir=out_model_dir,
            model_name=args.model_name,
            seed=seed,
            contrastive_weight=args.contrastive_weight,
            contrastive_temperature=args.contrastive_temperature,
            contrastive_eps=args.contrastive_eps,
            scheduler_eta_min=args.scheduler_eta_min,
            grad_clip_norm=args.grad_clip_norm,
            min_epochs=args.min_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            threshold_min_recall=args.threshold_min_recall,
            threshold_eps=args.threshold_eps,
            train_loader_shuffle=args.train_loader_shuffle,
            train_pin_memory=args.train_pin_memory,
            eval_pin_memory=args.eval_pin_memory
        )

        with open(log_fname, 'a') as f:
            f.write(
                '\n'.join(
                    ('-' * 25,
                    f'Fold {fold+1} Validation metrics:',
                    f'loss: {train_loss}',
                    f'Accuracies: {train_acc}',
                    f'Precisions: {train_p}',
                    f'Recalls: {train_r}',
                    f'F1s: {train_f1}',
                    f'AUCs: {train_auc}',
                    f'AUPRs: {train_aupr}',
                    '-' * 25 + '\n')
                )
            )
        plot_training_curves(
            train_loss,
            train_auc,
            save_dir=os.path.join(args.data_dir, dataset, f'figures_{args.log}', f'fold_{fold+1}')
        )

        test_acc, test_p, test_r, test_f1, test_auc, fpr, tpr, precision_curve, recall_curve, cm,test_aupr = model.eval_model(
            test_datasetmp_fold,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            flag=True,
            min_recall=args.threshold_min_recall,
            threshold_eps=args.threshold_eps,
            pin_memory=args.eval_pin_memory
        )
        fold_df = model.extract_raw_predictions(
            test_dataset=test_datasetmp_fold,
            fold=fold,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,  # 确保传入你在脚本中使用的 collate_fn
            pin_memory=args.eval_pin_memory
        )
        all_raw_predictions_dfs.append(fold_df)

        with open(log_fname, 'a') as f:
            f.write(
                '\n'.join(
                    ('-' * 25,
                    f'Fold {fold+1} Test metrics:',
                    f'Accuracy: {test_acc:.4f}',
                    f'Precision: {test_p:.4f}',
                    f'Recall: {test_r:.4f}',
                    f'F1: {test_f1:.4f}',
                    f'AUC: {test_auc:.4f}',
                    f'FPR: {fpr}\n',
                    f'TPR: {tpr}\n',
                    f'Precision curve: {precision_curve}\n',
                    f'Recall curve: {recall_curve}\n',
                    f'Confusion Matrix:\n{cm}',
                    f'APUR:\n{test_aupr}',
                    '-' * 25 + '\n')
                )
            )
        
        
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_prec.append(precision_curve)
        all_recall.append(recall_curve)
        all_cm.append(cm)
        all_auc.append(test_auc)
        all_aupr.append(test_aupr)
        plot_test_evaluation(
            fpr,
            tpr,
            precision_curve,
            recall_curve,
            cm,
            test_auc,
            save_dir=os.path.join(args.data_dir, dataset, f'figures_{args.log}', f'fold_{fold+1}')
        )

        fold_results.append({
            "train_acc": train_acc[-1],
            "train_precision": train_p[-1],
            "train_recall": train_r[-1],
            "train_f1": train_f1[-1],
            "train_auc": train_auc[-1],
            "test_acc": test_acc,
            "test_precision": test_p,
            "test_recall": test_r,
            "test_f1": test_f1,
            "test_auc": test_auc,
            "test_aupr":test_aupr
        })

    save_dir_base = os.path.join(args.data_dir, dataset, f'figures_{args.log}')
    os.makedirs(save_dir_base, exist_ok=True)
    
    final_predictions_df = pd.concat(all_raw_predictions_dfs, ignore_index=True)
    csv_path = os.path.join(save_dir_base, f'{args.experimental}_raw_predictions.csv')
    final_predictions_df.to_csv(csv_path, index=False)
    
    print(f"✅ 成功！所有折的 ROC 绘图原始数据已保存至: {csv_path}")
    logging.info(f"\n===== {args.n_splits}-Fold CV Summary =====")
    logging.info(args)
    logging.info(f"experiment: {args.experimental}")
    for metric in ['train_acc', 'train_precision', 'train_recall', 'train_f1', 'train_auc',
                'test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_auc','test_aupr']:
        values = [r[metric] for r in fold_results]
        logging.info(f"{metric} per fold: {['{:.4f}'.format(v) for v in values]}")
        logging.info(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    mean_fpr = np.linspace(0, 1, args.mean_curve_points)
    interp_tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)]
    mean_tpr = np.mean(interp_tprs, axis=0)
    std_tpr = np.std(interp_tprs, axis=0)   # ← 新增：标准差
    mean_tpr[-1] = 1.0

    mean_recall = np.linspace(0, 1, args.mean_curve_points)
    interp_precs = [np.interp(mean_recall, recall, prec) for recall, prec in zip(all_recall, all_prec)]
    mean_prec = np.mean(interp_precs, axis=0)
    std_prec = np.std(interp_precs, axis=0)  # ← 新增：标准差

    mean_auc = np.mean(all_auc)
    mean_cm = np.mean(np.stack(all_cm), axis=0)
    std_auc = np.std(all_auc) 
    plot_testcv_evaluation(
        mean_fpr, mean_tpr, std_tpr,
        mean_prec, std_prec, mean_recall,
        mean_cm,
        mean_auc,std_auc,
        save_dir=os.path.join(args.data_dir, dataset, f'figures_{args.log}', 'mean_summary')
    )
if __name__ == '__main__':
    args = parse_args()
    main(args)
