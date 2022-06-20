from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
import torch

BASE = "/home/jnascimento/exps/2022-7set-al/7Set-AL/"

def get_normalized_acc(y_true, y_pred):
    y_pred = y_pred.reshape([-1,1])
    norm_acc = []

    if torch.is_tensor(y_true):
        y_true = y_true.cpu()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu()

    for yt in np.unique(y_true):
        norm_acc.append(accuracy_score(y_true[y_true == yt], y_pred[y_true == yt]))
    return np.mean(norm_acc)


def get_f1(y_true, y_pred):
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
    

# def get_emb_vec(tensor_type=True):
#     df = pd.read_pickle('../../annotations/mobilnetV3_and_BERT_embeddings_dataframe.pkl').rename(columns={'txt': 'txt_label'})

#     N = len(df)

#     img_emb_size = len(df["img_embeddings"].iloc[0])
#     text_emb_size = len(df["tweet_embeddings"].iloc[0])

#     img_emb = torch.zeros([N,img_emb_size])
#     text_emb = torch.zeros([N,text_emb_size])
#     img_anno = torch.zeros([N], dtype=torch.int32)
#     text_anno = torch.zeros([N], dtype=torch.int32)

#     for i in range(N):
#         img_emb[i] = torch.Tensor(df["img_embeddings"].iloc[i])
#         text_emb[i] = torch.Tensor(df["tweet_embeddings"].iloc[i])
        
#         if df["img_label"].iloc[i] == "Relevant (Informative)":    
#             img_anno[i] = 1
#         else:
#             img_anno[i] = 0
            
#         if df["txt_label"].iloc[i] == "Relevant (Informative)":    
#             text_anno[i] = 1
#         else:
#             text_anno[i] = 0
    
#     if tensor_type is False:
#         return img_emb.numpy(), img_anno.numpy(), text_emb.numpy(), text_anno.numpy()
#     else:
#         return img_emb, img_anno, text_emb, text_anno


def get_emb_vec(tensor_type=True):

    out = dict()
    df = pd.read_pickle(BASE + 'annotations/mobilnetV3_and_BERT_embeddings_dataframe.pkl').rename(columns={'txt': 'txt_label'})

    out["img_emb"] = np.array(df["img_embeddings"].to_list()).astype(np.float32)
    out["text_emb"] = np.array(df["tweet_embeddings"].to_list()).astype(np.float32)

    N = len(df)

    out["img_anno"] = np.zeros(N, dtype=np.int32)
    out["text_anno"] = np.zeros(N, dtype=np.int32)

    out["train_mask"] = np.zeros(N, dtype=np.bool_)
    out["eval_mask"] = np.zeros(N, dtype=np.bool_)
    out["test_mask"] = np.zeros(N, dtype=np.bool_)

    ix = dict()

    for mode in ("train", "eval", "test"):
        s = pd.read_pickle(BASE + "artifacts/train_val_test_split/{}.pkl".format(mode))
    
        tweet_id_s = s['tweet_id'].to_numpy()

        s_n = len(tweet_id_s)

        ts = s["tweet_id"].to_numpy()
    
        df_s = df.query("tweet_id.isin(@ts)")

        ix[mode] = df_s.index.to_numpy()

        out["img_anno"][ix[mode]] = np.array(s["img_label"].to_list(), np.int32)
        out["text_anno"][ix[mode]] = np.array(s["txt"].to_list(), np.int32)
        
        out["{}_mask".format(mode)][ix[mode]] = True
    
    return out