from sklearn.metrics import f1_score, balanced_accuracy_score
import numpy as np
import pandas as pd
import torch

BASE = "/home/jnascimento/exps/2022-7set-al/7Set-AL/"

def get_normalized_acc(y_true, y_pred):
    if torch.is_tensor(y_true):
        y_true = y_true.cpu()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu()

    return balanced_accuracy_score(y_true, y_pred)


def get_f1(y_true, y_pred):
    if torch.is_tensor(y_true):
        y_true = y_true.cpu()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu()

    return f1_score(y_true, y_pred, average='weighted')


def plot_results(results_bacc, results_f1):
    bacc_mean = "BAcc - Mean: {}".format(np.mean(results_bacc, axis=0))
    bacc_std = "BAcc - Std: {}".format(np.std(results_bacc, axis=0))
    f1_mean = "F1 - Mean: {}".format(np.mean(results_f1, axis=0))
    f1_std = "F1 - Std: {}".format(np.std(results_f1, axis=0))

    print(bacc_mean)
    print(bacc_std)
    print(f1_mean)
    print(f1_std)
    

def get_emb_vec(emb):
    out = dict()

    df = pd.read_pickle(BASE + "artifacts/embeddings/tweet_data_all_emb.pkl")

    N = len(df)

    out["anno"] = np.zeros(N, dtype=np.int32)

    out["train_mask"] = np.zeros(N, dtype=np.bool_)
    out["eval_mask"] = np.zeros(N, dtype=np.bool_)
    out["test_mask"] = np.zeros(N, dtype=np.bool_)

    ix = dict()

    if emb == 'mobile':
        emb_mt = np.array(df["MobileNet_embeddings"].to_list()).astype(np.float32)
    elif emb == 'bert':
        emb_mt = np.array(df["Roberta_embeddings"].to_list()).astype(np.float32)
    elif emb == 'mobile+bert':
        emb_mt = np.array(df["MobileNet_Roberta_embeddings"].to_list()).astype(np.float32)
    elif emb == 'clipsum':
        emb_mt = np.array(df["clip_imgs_sum_text_embeddings"].to_list()).astype(np.float32)
    elif emb == 'clipcat':
        emb_mt = np.array(df["clip_imgs_cat_text_embeddings"].to_list()).astype(np.float32)

    out["emb_mt"] = emb_mt
    out['tweet_id'] = df['tweet_id'].to_numpy()

    for mode in ("train", "eval", "test"):
        s = pd.read_pickle(BASE + "artifacts/train_val_test_split/{}.pkl".format(mode))
    
        tweet_id_s = s['tweet_id'].to_numpy()

        s_n = len(tweet_id_s)

        ts = s["tweet_id"].to_numpy()

        df_s = df.query("tweet_id.isin(@ts)")

        ix[mode] = df_s.index.to_numpy()

        img_anno = np.array(s["img_label"].to_list(), np.int32)
        text_anno = np.array(s["txt"].to_list(), np.int32)

        out["anno"][ix[mode]] = img_anno | text_anno
        
        out["{}_mask".format(mode)][ix[mode]] = True
    
    return out
