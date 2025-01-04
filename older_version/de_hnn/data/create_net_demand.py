import sys
import json, gzip
import numpy as np
from scipy.sparse import coo_matrix 
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

abs_dir = '/data/zluo/'
rosettapath = abs_dir + 'new_data/'
raw_data_dir = rosettapath


designs_list = [
    'superblue1',
    'superblue2',
    'superblue3',
    'superblue4',
    'superblue5',
    'superblue6',
    'superblue7',
    'superblue18',
    'superblue19',
    'superblue9',
    'superblue11',
    'superblue14',
    'superblue16'
]
num_designs = len(designs_list)
num_variants_list = [
    5,
    5,
    6,
    5,
    6,
    6,
    6,
    5,
    6,
    6,
    6,
    6,
    6
]

assert num_designs == len(num_variants_list)

# Generate all names
sample_names = []
corresponding_design = []
corresponding_variant = []
for idx in range(num_designs):
    for variant in range(num_variants_list[idx]):
        sample_name = raw_data_dir + designs_list[idx] + '/' + str(variant + 1) + '/'
        sample_names.append(sample_name)
        corresponding_design.append(designs_list[idx])
        corresponding_variant.append(variant + 1)

# Synthetic data
N = len(sample_names)

for sample in range(50, 74):
    variant = corresponding_variant[sample]
    desname = corresponding_design[sample]
    print(desname, variant)

    data_dir = '2023-03-06_data/'

    print('Loading cell data')
    with gzip.open(f'{rosettapath}/cells.json.gz','rb') as f:
        cells = json.loads(f.read().decode('utf-8'))

    print('Loading instance and net data')
    with gzip.open(f'{rosettapath}/{desname}/{variant}/{desname}.json.gz','rb') as f:
        design = json.loads(f.read().decode('utf-8'))

    print('Loading congestion data')
    data=np.load(f'{rosettapath}/{desname}/{variant}/{desname}_congestion.npz')
    print('Flattening congestion data')
    demand=np.sum(data['demand'],axis=0)
    capacity=np.sum(data['capacity'],axis=0)

    def buildBST(array,start=0,finish=-1):
        if finish<0:
            finish = len(array)
        mid = (start + finish) // 2
        if mid-start==1:
            ltl=start
        else:
            ltl=buildBST(array,start,mid)
        if finish-mid==1:
            gtl=mid
        else:
            gtl=buildBST(array,mid,finish)
        return((array[mid],ltl,gtl))

    print('Building congestion index trees')
    xbst=buildBST(data['xBoundaryList'])
    ybst=buildBST(data['yBoundaryList'])

    def getGRCIndex(x, y, xbst, ybst):
        while (type(xbst)==tuple):
            if x < xbst[0]:
                xbst=xbst[1]
            else:
                xbst=xbst[2]
        while (type(ybst)==tuple):
            if y < ybst[0]:
                ybst=ybst[1]
            else:
                ybst=ybst[2]
        return ybst, xbst

    def getTermLoc(inst,termid,cells):
        orient=inst['orient']
        cell=cells[inst['cell']]
        if termid>len(cell['terms']) or termid<0:
            print(f"ERROR: inst {inst['id']} {inst['name']} term {termid} not found in cell {inst['cell']} {cell['name']}")
            term=cell['terms'][-1]
        else:
            term=cell['terms'][termid-1]
        if orient==0:   # R0
            x = inst['xloc']+term['xloc']
            y = inst['yloc']+term['yloc']
        elif orient==1: # R90
            x = inst['xloc']-term['yloc']
            y = inst['yloc']+term['xloc']
        elif orient==2: # R180
            x = inst['xloc']-term['xloc']
            y = inst['yloc']-term['yloc']
        elif orient==3: # R270
            x = inst['xloc']+term['yloc']
            y = inst['yloc']-term['xloc']
        elif orient==4: # MY
            x = inst['xloc']-term['xloc']
            y = inst['yloc']+term['yloc']
        elif orient==5: # MYR90
            x = inst['xloc']-term['yloc']
            y = inst['yloc']-term['xloc']
        elif orient==6: # MX
            x = inst['xloc']+term['xloc']
            y = inst['yloc']-term['yloc']
        elif orient==7: # MXR90
            x = inst['xloc']+term['yloc']
            y = inst['yloc']+term['xloc']
        else:
            raise Exception(f"Instance {inst['id']} {inst['name']} has unrecognized orientation {orient}")
        return x, y

    def findDuplicateConnData(conn):
        # Usage: dupconn, newconndata = findDuplicateConnData(conn)
        # where conn is the incidence matrix data loaded from
        # [design]_connectivity.npz .
        #
        # This procedure finds duplicate row and column entries 
        # in the incidence matrix and returns lists of the data
        # for those entries as dupconn[row][col].  It also 
        # returns an updated data array with each duplicate
        # entry set to -1.  The intent is that upon conversion
        # of a COO sparse matrix to CSC or CSR format, duplicate
        # indices will have data equal to -N_duplicates, which is
        # easy to identify and look up in dupconn.
        # 
        # An example usage of this code is with the superblue2 design, 
        # which has 1178 duplicate entries in the incidence matrix.
        # Consider net n791362 (id 762660), which has 2 connections 
        # to instance o921279 (id 897863)
        # of cell fake_macro_superblue2_o921279 (id 21218)
        # including terminals p53 (id 54) and p54 (id 55).
        # These appear as multiple entries 
        # for row 897863 col 762660 in the COO matrix.
        # Without this procedure, CSC matrix would sum both of
        # these entries, giving an incorrect termid of 109.
        # With this procedure, the CSC matrix entry is -2,
        # and dupconn[897863][762660] is [54, 55].
        #
        # Thanks to the following references that explained how to 
        # write this code:
        # https://urldefense.com/v3/__https://itecnote.com/tecnote/python-checking-for-and-indexing-non-unique-duplicate-values-in-a-numpy-array/__;!!Mih3wA!AjDqYbQ52HM6hdkTE-DPHJ3Vo5u94XI57a7vBStu42zUKkQnlI-ED4n8R7w22yelDq602ofEM0mfen5d0nT2$ 
        # https://urldefense.com/v3/__https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array__;!!Mih3wA!AjDqYbQ52HM6hdkTE-DPHJ3Vo5u94XI57a7vBStu42zUKkQnlI-ED4n8R7w22yelDq602ofEM0mferI2MuMk$ 
        #
        # Here is some explanation of the different arrays computed below:
        # struct   - array of all indices from the original connectivity arrays,
        #                organized in pairs for fast checking
        # uniq     - unique pairs of indices
        # uniq_cnt - number of entries in struct for each entry in uniq (same shape as uinq)
        # uniq_idx - corresponding index in uniq for each entry in struct (same shape as struct)
        # cnt_mask - True for each entry in uniq that has duplicates (same shape as uniq)
        # cnt_idx  - indices into uniq that have duplicates
        # idx_mask - True for each entry into struct that has duplicates (same shape as struct)
        # idx_idx  - indices into struct that have duplicates
        # uniq_idx[idx_mask] - indices in uniq that have duplicates (same shape as idx_idx)
        # srt_idx  - indices into uniq_idx[idx_mask] that would sort it
        # uniq_cnt[cnt_mask] - number of entries in struct for each pair of duplicate indices (same shape as cnt_idx)
        # dup_idx  - list of arrays with duplicate indices in struct

        # print('Building list of non-zero connectivity indices')
        connnzidx = np.stack((conn['row'], conn['col']),axis=1)
        # Some test data, for debugging, if necessary
        # connnzidx = np.array([[1,3],[2,2],[7,5],[2,2],[6,4],[4,9],[6,4],[5,8],[2,2]])
        ncols = connnzidx.shape[1]
        dtype = connnzidx.dtype.descr * ncols
        # print('Building struct')
        struct = connnzidx.view(dtype)

        # print('Finding unique entries')
        uniq, uniq_idx, uniq_cnt = np.unique(struct, return_inverse=True, return_counts=True)
        # print('Finding indices of duplicates')
        cnt_mask = uniq_cnt > 1
        cnt_idx, = np.nonzero(cnt_mask)
        idx_mask = np.in1d(uniq_idx, cnt_idx)
        idx_idx, = np.nonzero(idx_mask)
        srt_idx = np.argsort(uniq_idx[idx_mask])
        dup_idx = np.split(idx_idx[srt_idx], np.cumsum(uniq_cnt[cnt_mask])[:-1])

        # print('Copying connection data')
        newconndata=conn['data'].copy()
        # print('Building duplicate connection struct')
        dupconn={}
        for dia in dup_idx:
            instid, netid =struct[dia[0]][0]
            if instid not in dupconn:
                dupconn[instid]={}
            if netid not in dupconn[instid]:
                dupconn[instid][netid]=[]
            for di in dia:
                dupconn[instid][netid].append(newconndata[di])
                newconndata[di]=-1
        return dupconn, newconndata

    print('Loading connectivity data')
    conn=np.load(f'{rosettapath}/{desname}/{variant}/{desname}_connectivity.npz')
    print('Finding duplicate incidence matrix entries')
    dupconn, newconndata = findDuplicateConnData(conn)
    print('Building COO incidence matrix')
    coo = coo_matrix((newconndata, (conn['row'], conn['col'])), shape=conn['shape'])
    print('Converting to CSC incidence matrix')
    csc=coo.tocsc()

    # Good example of sparse row indexer:
    # https://urldefense.com/v3/__https://stackoverflow.com/questions/13843352/what-is-the-fastest-way-to-slice-a-scipy-sparse-matrix__;!!Mih3wA!AjDqYbQ52HM6hdkTE-DPHJ3Vo5u94XI57a7vBStu42zUKkQnlI-ED4n8R7w22yelDq602ofEM0mferNtFTnK$ 
    # Here we use a sparse column indexer in a similar style to
    # speed-up the traversal of the nets.  Nets are columns, so
    # the SciPy Compressed Sparse Column (csc) format is best.
    # This format claims to allow "efficient column slicing", which
    # is essentially what we are doing.  The csc object has 
    # the following attributes:
    #    data    - an array with all non-zero matrix entries
    #    indices - an array with the row indices associated with
    #              every value in the data array
    #    indptr  - an array of length Ncolumns+1, where indptr[i] 
    #              and indptr[i+1] give the start and stop
    #              indices into the data and indices arrays
    #              for column i


    congestion=np.zeros((len(design['nets']),3),dtype='int64')
    print('Building net-based congestion data list')
    for netid, col_start, col_end in tqdm(zip(range(csc.shape[1]),csc.indptr[:-1],csc.indptr[1:]),total=len(design['nets'])):
        xmax=None
        xmin=None
        ymax=None
        ymin=None
        degree=col_end-col_start
        for instid, incidencedata in zip(csc.indices[col_start:col_end],csc.data[col_start:col_end]):
            if incidencedata>=0:
                termidlist=[incidencedata]
            else:
                termidlist=dupconn[instid][netid]
            for termid in termidlist:
                x,y=getTermLoc(design['instances'][instid],termid,cells)
                if xmax==None:
                    xmax=x
                    xmin=x
                    ymax=y
                    ymin=y
                else:
                    xmax=max(xmax,x)
                    xmin=min(xmin,x)
                    ymax=max(ymax,y)
                    ymin=min(ymin,y)
        net=design['nets'][netid]
        net['degree']=degree
        if xmax!=None:
            i1,j1=getGRCIndex(xmin,ymin,xbst,ybst)
            i2,j2=getGRCIndex(xmax,ymax,xbst,ybst)
            dem=np.sum(demand[i1:i2+1,j1:j2+1])
            cap=np.sum(capacity[i1:i2+1,j1:j2+1])
            grccount=(i2+1-i1)*(j2+1-j1)
            congestion[netid]=(dem,cap,grccount)

    # nonzeronets=congestion[:,2]!=0
    # avgdemand=congestion[nonzeronets,0]/congestion[nonzeronets,2]
    # # avgcapacity=congestion[nonzeronets,1]/congestion[nonzeronets,2]
    # plt.hist(avgdemand,bins=100)
    # plt.title(f"Design {desname} variant {variant}")
    # plt.xlabel('Average demand')
    # plt.ylabel('Number of nets')
    # plt.show()

    avg_demand = []
    avg_capacity = []
    for item in congestion:
        if np.any(item == 0):
            avg_demand.append(0)
            avg_capacity.append(0)
        else:
            avg_demand.append(item[0]/item[2])
            avg_capacity.append(item[1]/item[2])
    
    dictionary = {
        "demand": avg_demand, 
        "capacity": avg_capacity
    }
    
    fn = data_dir + '/' + str(sample) + '.net_demand_capacity.pkl'
    f = open(fn, "wb")
    pickle.dump(dictionary, f)
    f.close()
    print('Save file', fn)

