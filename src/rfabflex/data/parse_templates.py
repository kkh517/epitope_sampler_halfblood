import re
from collections import namedtuple
import torch
import numpy as np
from ffindex import read_index, read_data, get_entry_by_name, read_entry_lines
import util
import argparse
from pathlib import Path

def parse_pdb_lines_w_seq(lines):

    """Read and extract xyz coords of N, Ca, C atoms from a PDB file

    Args:
        lines (list): list of lines of PDB file

    Returns:
        xyz:  xyz coordinate of pdb file    [L, 14, 3] (14: 4BB + up to 10 SC atoms)
        mask: mask for atom existence       [L, 14]
        np.array(idx_s): residue number     [L]
        np.array(seq) : sequence            [L]
    """

    res = [
        (l[22:26], l[17:20])
        for l in lines
        if l[:4] == "ATOM" and l[12:16].strip() == "CA"
    ]
    idx_s = [int(r[0]) for r in res]
    seq = [util.aa2num[r[1]] if r[1] in util.aa2num.keys() else 20 for r in res]

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(idx_s), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        resNo, atom, aa = int(l[22:26]), l[12:16], l[17:20]
        idx = idx_s.index(resNo)
        for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
            if tgtatm == atom and i_atm < 14:
                xyz[idx, i_atm, :] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[..., 0]))
    xyz[np.isnan(xyz[..., 0])] = 0.0

    return xyz, mask, np.array(idx_s), np.array(seq)

def parse_templates(params):

    """Parse templates from paramters

    Args:
        parameters (dictionary)
        params['FFDB']  : FFDB location
        params['hhr']   : hhr file location
        params['atab']  : atab file location

    Returns:
        xyz:  xyz coordinate of templates           [N_templ * L_align, 14, 3]
        mask: mask for atom existence               [N_templ * L_align, 14]
        qmap: aligned part of query, template no    [N_templ * L_align, 2]
        f0d: template properties                    [N_templ, 4]
             (Identities, Similarity, Sum_probs, Template_Neff)
        f1d: template properties per residue        [N_templ * L_align, 3]
             (Score, SS, prob)
        ids: template pdb id                        [N_templ]
        seq: template sequence                      [N_templ * L_align]
    """

    # init FFindexDB of templates
    ### and extract template IDs
    ### present in the DB
    #print('parse_templates', params['atab'])
    

    FFindexDB = namedtuple("FFindexDB", "index, data")
    ffdb = FFindexDB(
        read_index(params["FFDB"] + "_pdb.ffindex"),
        read_data(params["FFDB"] + "_pdb.ffdata"),
    )
    

    # ffids = set([i.name for i in ffdb.index])

    # process tabulated hhsearch output to get
    # matched positions and positional scores
    
    hits = []
    for l in open(params["atab"], "r", encoding="utf-8").readlines():
        if l[0] == ">":
            key = l[1:].split()[0]
            hits.append([key, [], []])
        elif "score" in l or "dssp" in l:
            continue
        else:
            hi = l.split()[:5] + [0.0, 0.0, 0.0]
            hits[-1][1].append([int(hi[0]), int(hi[1])])
            hits[-1][2].append([float(hi[2]), float(hi[3]), float(hi[4])])
    
    hits = hits[:50]

    # get per-hit statistics from an .hhr file
    # (!!! assume that .hhr and .atab have the same hits !!!)
    # [Probab, E-value, Score, Aligned_cols,
    # Identities, Similarity, Sum_probs, Template_Neff]

    lines = open(params["hhr"], "r", encoding="utf-8").readlines()
    pos = [i + 1 for i, l in enumerate(lines) if l[0] == ">"][:50]
    for i, posi in enumerate(pos):
        hits[i].append(
            [float(s) for s in re.sub("[=%]", " ", lines[posi]).split()[1::2]]
        )

    # parse templates from FFDB
    #print(len(hits))
    for hi in hits:
        # if hi[0] not in ffids:
        #    continue
        #ffdb_start_time = time.time()
        entry = get_entry_by_name(hi[0], ffdb.index)
        
       
        if entry is None:
            continue
        data = read_entry_lines(entry, ffdb.data)

        
        hi += list(parse_pdb_lines_w_seq(data))



    # process hits
    counter = 0
    xyz, qmap, mask, f0d, f1d, ids, seq = [], [], [], [], [], [], []
    for data in hits:
        if len(data) < 7:
            continue

        qi, ti = np.array(data[1]).T
        _, sel1, sel2 = np.intersect1d(ti, data[6], return_indices=True)
        ncol = sel1.shape[0]
        #if ncol < 10:
        #    continue

        ids.append(data[0])
        f0d.append(data[3])
        f1d.append(np.array(data[2])[sel1])
        xyz.append(data[4][sel2])
        #print(torch.from_numpy(data[4][sel2].astype(np.float32)).size())
        mask.append(data[5][sel2])
        seq.append(data[-1][sel2])
        qmap.append(np.stack([qi[sel1] - 1, [counter] * ncol], axis=-1))
        counter += 1

    ids = ids
    xyz = np.vstack(xyz).astype(np.float32)
    mask = np.vstack(mask).astype(bool)
    #print('tplt_mask', list(mask))
    qmap = np.vstack(qmap).astype(np.compat.long)
    f0d = np.vstack(f0d).astype(np.float32)
    f1d = np.vstack(f1d).astype(np.float32)
    seq = np.hstack(seq).astype(np.compat.long)
    
    #print(torch.from_numpy(xyz).size())
    #print(torch.from_numpy(mask).size())
    #print(torch.from_numpy(qmap).size())
    #print(torch.from_numpy(f0d).size())
    #print(torch.from_numpy(f1d).size())
    #print(len(ids))
    #print(torch.from_numpy(seq).size())
    return {
        "xyz": torch.from_numpy(xyz),
        "mask": torch.from_numpy(mask),
        "qmap": torch.from_numpy(qmap),
        "f0d": torch.from_numpy(f0d),
        "f1d": torch.from_numpy(f1d),
        "ids": ids,
        "seq": torch.from_numpy(seq),
    }

if __name__ == '__main__':
    
    params = {}
    parser = argparse.ArgumentParser(description='Parse templates')
    FFDB = "/home/yubeen/RF2/pdb100_2021Mar03/pdb100_2021Mar03"
    parser.add_argument('atab', help = 'location of atab file')
    parser.add_argument('hhr', help='location of hhr file')
    args = parser.parse_args()
    
    params['FFDB'] = FFDB
    params['atab'] = args.atab
    params['hhr'] = args.hhr
    
    directory = Path(args.atab).parent
    torch.save(parse_templates(params), f'{directory}/templ_info.pt')
    
