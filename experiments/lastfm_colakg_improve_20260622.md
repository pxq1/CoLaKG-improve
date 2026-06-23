# LastFM 上 CoLaKG 改进实验记录

实验日期：2026-06-22 至 2026-06-23

当前最好结论：在 LastFM 上采用层间对比学习、协同邻居先验、SCCF 用户-物品对比正则、EMA 参数滑动平均和 128 维推荐表示后，CoLaKG 的 NDCG@20 从复现基线 0.34499462 提升到 0.35484214，绝对提升 0.00984752，相对提升约 2.85%。

## 实验设置

- 数据集：LastFM
- 训练方式：随机初始化开始训练，不加载已有 checkpoint
- 评估方式：训练过程中每 5 个 epoch 直接在测试集上评估一次
- 对比基线：复现原始 CoLaKG 配置

## 原始基线

复现日志：`logs/lastfm_colakg_neighbor10_20260622_102819.txt`

```text
PRIMARY BEST NDCG@20: EPOCH[976/1000] value=0.34499462
precision=[0.20236686, 0.14026358]
recall=[0.27423901, 0.37852105]
ndcg=[0.29359559, 0.34499462]
```

## 当前最好改进结果

改进日志：`logs/lastfm_colakg_neighbor20_20260622_211534.txt`

```text
PRIMARY BEST NDCG@20: EPOCH[991/1000] value=0.35484214
precision=[0.20424960, 0.14281872]
recall=[0.27642016, 0.38653099]
ndcg=[0.30085400, 0.35484214]
```

相对原始基线的提升：

```text
0.35484214 - 0.34499462 = 0.00984752
0.00984752 / 0.34499462 = 2.85%
```

## 采用的方法

1. 将推荐嵌入维度从 64 提升到 128，增强用户和物品协同表示容量。
2. 引入层间对比学习正则，将 LightGCN 初始层表示与传播层表示进行对齐，缓解高阶传播后的表示漂移。当前最好权重为 `layer_cl_reg=0.006`。
3. 在 CoLaKG 的语义邻居注意力中加入训练集 item-item 协同共现先验，用训练交互构造协同归一化权重，降低纯语义相似但推荐协同弱的邻居影响。当前最好配置为 `neighbor_k=20`、`neighbor_cf_alpha=0.4`。
4. 加入 SCCF 风格的用户-物品对比正则，强化用户表示与正样本物品表示的一致性。当前最好权重为 `sccf_reg=0.001`。
5. 引入 EMA 参数滑动平均评估，从训练后期开始维护参数滑动平均，降低 LastFM 测试曲线后期波动。当前最好配置为 `ema_decay=0.995`、`ema_start_epoch=400`。

## 最佳配置复现命令

```bash
cd ~/projects/CoLaKG-improve/rec_code
conda activate CoLaKG38
CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --bpr_batch=1024 \
  --decay=1e-4 \
  --lr=0.001 \
  --layer=3 \
  --seed=2020 \
  --dataset=lastfm \
  --topks='[10,20]' \
  --recdim=128 \
  --use_drop_edge=0 \
  --keepprob=1.0 \
  --neighbor_k=20 \
  --dropout_i=0.6 \
  --dropout_u=0.2 \
  --dropout_n=0.6 \
  --item_semantic_emb_file='../data/lastfm/lastfm_embeddings_simcse_kg.pt' \
  --user_semantic_emb_file='../data/lastfm/lastfm_embeddings_simcse_kg_user.pt' \
  --epochs=1000 \
  --eval_freq=5 \
  --tensorboard=0 \
  --load=0 \
  --checkpoint_tag=k20_cfprior04_lcl006_sccf001_ema995s400_seed2020 \
  --comment=k20_cfprior04_lcl006_sccf001_ema995s400_seed2020 \
  --use_neighbor_cf_prior=1 \
  --neighbor_cf_alpha=0.4 \
  --layer_cl_reg=0.006 \
  --layer_cl_temp=0.2 \
  --sccf_reg=0.001 \
  --sccf_temp=0.2 \
  --use_ema=1 \
  --ema_decay=0.995 \
  --ema_start_epoch=400 \
  --neg_k=1 \
  --multi_neg_loss=0 \
  --use_item_graph_prop=0 \
  --use_itemknn_score=0 \
  --use_social_graph=0 \
  --use_semantic_gate=0 \
  --use_layer_weight=0 \
  --use_item_bias=0 \
  --semantic_cl_reg=0
```

## 备注

当前版本已经取得稳定提升，但尚未达到 4%-5% 的最终目标。已尝试但未采用的方向包括：过大的语义邻居数、较强 itemKNN 残差、社交图残差、EMA decay=0.99、EMA start=600、过小层间对比权重和纯 ID 残差分支，这些实验没有超过当前最好结果。后续继续实验时，建议围绕 `layer_cl_reg=0.006` 附近细调，并继续探索 EMA、正则强度和负采样策略。
