from __future__ import annotations
from pathlib import Path
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import scipy.sparse as sp
from scipy.sparse import csr_matrix, triu


############################# General Settings #############################
random_seed = 43

root = '/home/qingyuyang/ESM4SL/data'

with open(f'{root}/mapping/name2id.pkl', 'rb') as f1:
    name2id = pickle.load(f1)  # Len = 27671.  key: gene name, value: gene id, both str.

with open(f'{root}/mapping/gid2uid.pkl', 'rb') as f2:
    gid2uid = pickle.load(f2)  # Len = 9845. key: entrez id, value: unified id in SLBench, both int.


############################# SLKB Cleaning and Preprocessing #############################
def count_pn_ratio(df: pd.DataFrame) -> tuple[int, int, int, float]:
    total = len(df)
    pos = len(df[df['2'] == 1].index.tolist())
    neg = len(df[df['2'] == 0].index.tolist())
    if pos != 0:
        ratio = round(neg / pos, 4)
    else:
        ratio = None
    return total, pos, neg, ratio


def select_cl_from_slkb(df: pd.DataFrame, cl: str) -> pd.DataFrame:
    cl_df = df[df['3'] == cl]
    cl_df.reset_index(drop=True, inplace=True)
    return cl_df


def remove_rows_with_condition(df: pd.DataFrame) -> pd.DataFrame:
    duplicates = df[df.duplicated(subset=['0', '1', '3'], keep=False)]
    for _, group in duplicates.groupby(['0', '1', '3']):
        assert len(group['2'].unique()) == len(group), f"Pair ({group.iloc[0]['0']}, {group.iloc[0]['1']}) have duplicate same labels!"
        df.drop(group.index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def format_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[['g1', 'g2', 'label', 'cell_line']]
    df.columns = ['0', '1', '2', '3']
    df.drop_duplicates(inplace=True, ignore_index=True)
    return df


def df_gene2id(df: pd.DataFrame) -> pd.DataFrame:
    df['0'] = df['0'].map(name2id)
    df['1'] = df['1'].map(name2id)
    df = df.dropna()  # NOTE: some gene may not have id mappings
    df.reset_index(drop=True, inplace=True)
    df['0'] = df['0'].astype(int)
    df['1'] = df['1'].astype(int)
    return df


def filter_baseline_genes(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['0'].isin(gid2uid.keys()) & df['1'].isin(gid2uid.keys())]
    df.reset_index(drop=True, inplace=True)
    return df


def count_gene_freq(df: pd.DataFrame) -> tuple[list[int], list[int], dict[int, int]]:
    gene_list = list(df['0']) + list(df['1'])
    gene_set = list(set(gene_list))

    gene_count = {}
    for gene in gene_set:
        gene_count[gene] = gene_list.count(gene)
    return gene_list, gene_set, gene_count


def visualize_gene_freq(gene_count: dict, max_gene_show: int = 200, show_yval_step: int = 10, show_xlabel_step: int = 5, 
                        plot: bool = False, save_path: Path | None = None, title: str | None = None):
    gene_num = len(gene_count)
    jump = int(gene_num / max_gene_show) if gene_num >= max_gene_show else 1
    sorted_list = sorted(gene_count.items(), key=lambda item: item[1], reverse=True)[::jump]
    keys, values = zip(*sorted_list)
    keys = [str(i) for i in keys]

    fig, ax = plt.subplots(figsize=(20, 8))
    bars = ax.bar(keys, values)
    for i, bar in enumerate(bars):
        if i % show_yval_step == 0:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')
    if title:
        plt.title(f"{title}")
    plt.xlabel('Genes')
    plt.ylabel('Number of appearances in SL pairs')
    plt.xticks(np.arange(len(keys))[::show_xlabel_step], keys[::show_xlabel_step], rotation=90)
    if save_path:
        plt.savefig(f"{save_path}/{title}.jpg")
    if not plot:
        plt.clf()
        plt.close(fig)
    else:
        plt.show()


def construct_mat(df: pd.DataFrame, gene_set: list[str] | None = None) -> sp.csr_matrix:
    gene_set = list(set(df['0']) |  set(df['1']))
    num_node = len(gene_set)
    unified_id = dict(zip(gene_set, range(num_node)))
    new_df = df.copy()
    new_df['0'] = new_df['0'].map(unified_id).astype(int)
    new_df['1'] = new_df['1'].map(unified_id).astype(int)

    position = np.asarray(new_df[['0', '1']].values, dtype=int)
    for i in range(len(position)):
        if position[i, 0] > position[i, 1]:
            position[i, 0], position[i, 1] = position[i, 1], position[i, 0]

    return sp.csr_matrix((np.ones(len(position)), (position[:, 0], position[:, 1])), shape=(num_node, num_node), dtype='bool')


def is_upper_triangle_filled(matrix: csr_matrix) -> bool:
    if not isinstance(matrix, csr_matrix):
        raise ValueError("Input matrix must be scipy.sparse.csr_matrix!")
    upper_triangle = triu(matrix, k=1)  # k=1 表示排除对角线
    non_zero_values = upper_triangle.data
    return len(non_zero_values) == matrix.shape[0] * (matrix.shape[0] - 1)


def arrays_equal(arr1: np.ndarray, arr2: np.ndarray) -> bool:
    return set(map(tuple, arr1)) == set(map(tuple, arr2))


def are_dataframes_equal(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    return (not df1.duplicated().any()) and len(df1) == len(df2) and set(tuple(x) for x in df1.values) == set(tuple(x) for x in df2.values)


def are_dataframes_subset(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    return set(tuple(x) for x in df1.values) <= set(tuple(x) for x in df2.values)


def split_cv(df: pd.DataFrame, save_path: Path, num_splits: int = 5):
    os.makedirs(save_path, exist_ok=True)

    kf = KFold(n_splits=num_splits, shuffle=True, random_state=random_seed)
    for i, (train_index, test_index) in enumerate(kf.split(df)):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

        assert are_dataframes_equal(pd.concat([train_df, test_df], ignore_index=True), df)

        train_df.to_csv(f'{save_path}/sl_train_{i}.csv', index=False)
        test_df.to_csv(f'{save_path}/sl_test_{i}.csv', index=False)


def check_C2_property(test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    train_genes = list(set(train_df['0']) | set(train_df['1']))
    test_gene_pairs = list(zip(list(test_df['0']), list(test_df['1'])))

    for g1, g2 in test_gene_pairs:
        g1_freq = train_genes.count(g1)
        g2_freq = train_genes.count(g2)
        if not ((g1_freq > 0 and g2_freq == 0) or (g1_freq == 0 and g2_freq > 0)):
            index_names = test_df[(test_df['0'] == g1) & (test_df['1'] == g2)].index
            test_df.drop(index_names, inplace=True)

    return test_df


def get_pairs(sl_pairs: np.ndarray[int], train_genes: np.ndarray[int], test_genes: np.ndarray[int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    train_pairs = []
    test_pairs = []

    for pair in sl_pairs:
        if pair[0] in train_genes and pair[1] in train_genes:
            train_pairs.append(list(pair))
        elif (pair[0] in test_genes and pair[1] in train_genes) or (pair[0] in train_genes and pair[1] in test_genes):
            test_pairs.append(list(pair))

    return train_pairs, test_pairs


def C2(df: pd.DataFrame, save_path: Path, fold_num: int = 5) -> None:
    os.makedirs(save_path, exist_ok=True)

    sl_pairs = df[['0', '1']].to_numpy()
    gene1 = set(df['0'])
    gene2 = set(df['1'])
    genes = np.array(list(gene1 | gene2))

    kf = KFold(n_splits=fold_num, shuffle=True, random_state=random_seed)
    for i, (train_index, test_index) in enumerate(kf.split(genes)):
        train_genes = genes[train_index]
        test_genes = genes[test_index]

        train_pairs, test_pairs = get_pairs(sl_pairs, train_genes, test_genes)

        train_pairs_df = pd.DataFrame(train_pairs, columns=['0', '1'])
        result_train_df = pd.merge(df, train_pairs_df, on=['0', '1'], how='inner')
        test_pairs_df = pd.DataFrame(test_pairs, columns=['0', '1'])
        result_test_df = pd.merge(df, test_pairs_df, on=['0', '1'], how='inner')

        assert are_dataframes_subset(pd.concat([result_train_df, result_test_df], ignore_index=True), df)
        result_test_df = check_C2_property(result_test_df, result_train_df)

        result_train_df.to_csv(f'{save_path}/sl_train_{i}.csv', index=False)
        result_test_df.to_csv(f'{save_path}/sl_test_{i}.csv', index=False)


def form_tail_df(cl_df: pd.DataFrame) -> pd.DataFrame:
    _, _, cl_count = count_gene_freq(cl_df)
    cl_tail_nodes = [gene for gene, freq in cl_count.items() if freq == 1]
    cl_tail_df = cl_df[cl_df['0'].isin(cl_tail_nodes) & cl_df['1'].isin(cl_tail_nodes)]
    # print(f"{len(cl_tail_nodes)} tail nodes, {len(cl_tail_df)} tail pairs")
    return cl_tail_df


def tail(df: pd.DataFrame, save_path: Path, test_rate: float = 0.2, min_tail_rate: float = 0.02, fold_num: int = 5) -> None:
    os.makedirs(save_path, exist_ok=True)

    for i in range(fold_num):
        tail_df = form_tail_df(df).sample(frac=1)
        if len(tail_df) <= int(min_tail_rate * len(df)):
            print('This cell line is unsuitable for long-tail scene!')
            os.remove(save_path)
            return None

        target_len = int(test_rate * len(df))
        test_df = tail_df[:target_len] if len(tail_df) > target_len else tail_df  # select some tails to form test set

        train_df = df.merge(test_df, how='left', indicator=True)  # train df = whole df - test df
        train_df = train_df[train_df['_merge'] == 'left_only'][['0', '1', '2', '3']]

        assert are_dataframes_equal(pd.concat([train_df, test_df], ignore_index=True), df)

        train_df.to_csv(f'{save_path}/sl_train_{i}.csv', index=False)
        test_df.to_csv(f'{save_path}/sl_test_{i}.csv', index=False)


def count_specific_statistics(table_save_path: Path) -> pd.DataFrame:
    column_names = ['cell_line', 'scene', 'fold', 'train(1)/test(0)', '# pairs', '# pos', '# neg', 'n/p ratio', '# genes', '# unique genes']
    datasets_stat = pd.DataFrame(columns=column_names)

    cell_lines = [i for i in os.listdir(table_save_path) if not i.endswith('.csv')]

    for cl, scene in tqdm(itertools.product(cell_lines, ['C1', 'C2', 'Tail'])):
        path = f'{table_save_path}/{cl}/{scene}'
        if not os.path.exists(path) or not len(os.listdir(path)):
            continue
        for fold in range(5):
            train_df = pd.read_csv(f'{path}/sl_train_{fold}.csv')
            total, pos, neg, ratio = count_pn_ratio(train_df)
            gene_list, gene_set, gene_count = count_gene_freq(train_df)
            unique_gene_num = list(gene_count.values()).count(1)
            datasets_stat.loc[len(datasets_stat)] = [cl, scene, fold, 1, total, pos, neg, ratio, len(gene_set), unique_gene_num]

            test_df = pd.read_csv(f'{path}/sl_test_{fold}.csv')
            total, pos, neg, ratio = count_pn_ratio(test_df)
            gene_list, gene_set, gene_count = count_gene_freq(test_df)
            unique_gene_num = list(gene_count.values()).count(1)
            datasets_stat.loc[len(datasets_stat)] = [cl, scene, fold, 0, total, pos, neg, ratio, len(gene_set), unique_gene_num]

    datasets_stat.to_csv(f'{table_save_path}/datasets_stat.csv', index=False)
    return datasets_stat


def form_id_seq_list(gids: set[int], mapping: pd.DataFrame, max_len: int = 2000) -> list[tuple[int, str]]:
    id_seq = []
    for gid in gids:
        seq = mapping[mapping['From'] == gid]['Sequence'].values[0]
        if len(seq) > max_len:
            seq = seq[:max_len]
        id_seq.append((gid, seq))
    return id_seq


############################# Baseline Preprocessing #############################
def df_gid2uid(df: pd.DataFrame) -> pd.DataFrame:
    df['0'] = df['0'].map(gid2uid)
    df['1'] = df['1'].map(gid2uid)
    df = df.dropna()  # NOTE: some gene may not have mappings
    df['0'] = df['0'].astype(int)
    df['1'] = df['1'].astype(int)
    return df


def gid2uid_all(data_path: Path, save_path: Path, fold: int = 5, if_transfer: bool = False) -> None:
    os.makedirs(save_path, exist_ok=True)

    for i in range(fold):
        fn_num = 0 if if_transfer else i

        train_df_gid = pd.read_csv(f'{data_path}/sl_train_{fn_num}.csv')
        train_df_uid = df_gid2uid(train_df_gid)
        assert len(train_df_uid) == len(train_df_gid)
        train_df_uid.to_csv(f'{save_path}/sl_train_{fn_num}.csv', index=False)

        test_df_gid = pd.read_csv(f'{data_path}/sl_test_{fn_num}.csv')
        test_df_uid = df_gid2uid(test_df_gid)
        assert len(test_df_uid) == len(test_df_gid)
        test_df_uid.to_csv(f'{save_path}/sl_test_{fn_num}.csv', index=False)


def construct_data_npy(data_path: Path, save_path: Path, fold: int = 5, if_transfer: bool = False) -> None:
    sl_pos_train = []
    sl_pos_test = []
    sl_neg_train = []
    sl_neg_test = []

    for i in range(fold):
        fn_num = 0 if if_transfer else i

        sl_data_train = pd.read_csv(os.path.join(data_path, f'sl_train_{fn_num}.csv'))
        sl_data_test = pd.read_csv(os.path.join(data_path, f'sl_test_{fn_num}.csv'))

        sl_data_pos_train = sl_data_train[['0', '1']][sl_data_train['2'] == 1].to_numpy()
        sl_data_pos_test = sl_data_test[['0', '1']][sl_data_test['2'] == 1].to_numpy()
        sl_data_neg_train = sl_data_train[['0', '1']][sl_data_train['2'] == 0].to_numpy()
        sl_data_neg_test = sl_data_test[['0', '1']][sl_data_test['2'] == 0].to_numpy()

        sl_pos_train.append(sl_data_pos_train)
        sl_pos_test.append(sl_data_pos_test)
        sl_neg_train.append(sl_data_neg_train)
        sl_neg_test.append(sl_data_neg_test)

    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'pos_train.npy'), sl_pos_train)
    np.save(os.path.join(save_path, 'pos_test.npy'), sl_pos_test)
    np.save(os.path.join(save_path, 'neg_train.npy'), sl_neg_train)
    np.save(os.path.join(save_path, 'neg_test.npy'), sl_neg_test)


def extract_sparse_mat_from_df(df_path: Path) -> tuple[csr_matrix, csr_matrix]:
    df = pd.read_csv(df_path)
    pos = np.array(df[df['2'] == 1])
    neg = np.array(df[df['2'] == 0])
    len_ = len(pos)
    len_neg = len(neg)
    sparse_matrix_pos = csr_matrix((np.ones(len_), (pos[:, 0], pos[:, 1])), shape=(9845, 9845)) 
    sparse_matrix_neg = csr_matrix((np.ones(len_neg), (neg[:, 0], neg[:, 1])), shape=(9845, 9845))
    return sparse_matrix_pos, sparse_matrix_neg


def construct_data_ptgnn(data_path: Path, save_path: Path = None, fold: int = 5, if_transfer: bool = False) -> None:        
    graph_train_pos_kfold = []
    graph_test_pos_kfold = []
    graph_train_neg_kfold = []
    graph_test_neg_kfold = []

    for i in range(fold):
        fn_num = 0 if if_transfer else i

        sparse_matrix_pos, sparse_matrix_neg = extract_sparse_mat_from_df(os.path.join(data_path, f'sl_train_{fn_num}.csv'))
        graph_train_pos_kfold.append(sparse_matrix_pos)
        graph_train_neg_kfold.append(sparse_matrix_neg)

        sparse_matrix_pos, sparse_matrix_neg = extract_sparse_mat_from_df(os.path.join(data_path, f'sl_test_{fn_num}.csv'))
        graph_test_pos_kfold.append(sparse_matrix_pos)
        graph_test_neg_kfold.append(sparse_matrix_neg)

    pos_graph = [graph_train_pos_kfold, graph_test_pos_kfold]
    neg_graph = [graph_train_neg_kfold, graph_test_neg_kfold] 

    os.makedirs(save_path, exist_ok=True)
    np.save(f"{save_path}/pos_graph.npy", pos_graph)
    np.save(f"{save_path}/neg_graph.npy", neg_graph)

