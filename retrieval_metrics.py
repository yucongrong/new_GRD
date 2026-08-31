import numpy as np


def compute_ap(ranks, nres):
    nimgranks = len(ranks)
    ap = 0
    recall_step = 1. / nres
    for j in np.arange(nimgranks):
        rank = ranks[j]
        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank
        precision_1 = float(j + 1) / (rank + 1)
        ap += (precision_0 + precision_1) * recall_step / 2.
    return ap


def compute_map(ranks, gnd, kappas=[]):
    map = 0.
    nq = len(gnd)
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0
    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue
        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)
        pos = np.arange(ranks.shape[0])[np.isin(ranks[:, i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.isin(ranks[:, i], qgndj)]
        k = 0
        ij = 0
        if len(junk):
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap
        pos += 1
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]
    map = map / (nq - nempty)
    pr = pr / (nq - nempty)
    return map, aps, pr, prs


def evaluate_setup(ranks, gnd_data, setup_type):
    ks = [1, 5, 10]
    ranks = ([[ranks[row][col] for row in range(len(ranks))] for col in range(len(ranks[0]))])
    ranks = np.array(ranks)
    gnd = [item.copy() for item in gnd_data]
    for item in gnd:
        if setup_type == 'E':
            item['ok'] = item['easy'].copy()
            item['junk'] = np.concatenate([item['junk'], item['hard']]) if 'hard' in item else item['junk'].copy()
        elif setup_type == 'M':
            item['ok'] = np.concatenate([item['easy'], item['hard']]) if 'hard' in item else item['easy'].copy()
            item['junk'] = item['junk'].copy()
        elif setup_type == 'H':
            item['ok'] = item['hard'].copy() if 'hard' in item else np.array([])
            item['junk'] = np.concatenate([item['junk'], item['easy']]) if 'easy' in item else item['junk'].copy()
        else:
            raise ValueError(f"不支持的评估类型：{setup_type}，仅支持'E'/'M'/'H'")
    map_score, aps, mpr, prs = compute_map(ranks, gnd, ks)
    return map_score, aps, mpr, prs
