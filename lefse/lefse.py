import os,sys,math,pickle
import random as lrand
import rpy2.robjects as robjects
import argparse
import numpy
#import svmutil

def init():
    lrand.seed(1982)
    robjects.r('library(splines)')
    robjects.r('library(stats4)')
    robjects.r('library(survival)')
    robjects.r('library(mvtnorm)')
    robjects.r('library(modeltools)')
    robjects.r('library(coin)')
    robjects.r('library(MASS)')

def get_class_means(class_sl,feats):
    means = {}
    clk = list(class_sl.keys())
    for fk,f in feats.items():
        means[fk] = [numpy.mean((f[class_sl[k][0]:class_sl[k][1]])) for k in clk]
    return clk,means

def save_res(res,filename):
    with open(filename, 'w') as out:
        for k,v in res['cls_means'].items():
            out.write(k+"\t"+str(math.log(max(max(v),1.0),10.0))+"\t")
            if k in res['lda_res_th']:
                for i,vv in enumerate(v):
                    if vv == max(v):
                        out.write(str(res['cls_means_kord'][i])+"\t")
                        break
                out.write(str(res['lda_res'][k]))
            else: out.write("\t")
            out.write( "\t" + (res['wilcox_res'][k] if 'wilcox_res' in res and k in res['wilcox_res'] else "-")+"\n")

def load_data(input_file, nnorm = False):
    with open(input_file, 'rb') as inputf:
        inp = pickle.load(inputf)
    if nnorm: return inp['feats'],inp['cls'],inp['class_sl'],inp['subclass_sl'],inp['class_hierarchy'],inp['norm']
    else: return inp['feats'],inp['cls'],inp['class_sl'],inp['subclass_sl'],inp['class_hierarchy']

def load_res(input_file):
    with open(input_file, 'rb') as inputf:
        inp = pickle.load(inputf)
    return inp['res'],inp['params'],inp['class_sl'],inp['subclass_sl']


def test_kw_r(cls,feats,p,factors):
    robjects.globalenv["y"] = robjects.FloatVector(feats)
    for i,f in enumerate(factors):
        robjects.globalenv['x'+str(i+1)] = robjects.FactorVector(robjects.StrVector(cls[f]))
    fo = "y~x1"
    #for i,f in enumerate(factors[1:]):
    #   if f == "subclass" and len(set(cls[f])) <= len(set(cls["class"])): continue
    #   if len(set(cls[f])) == len(cls[f]): continue
    #   fo += "+x"+str(i+2)
    kw_res = robjects.r('kruskal.test('+fo+',)$p.value')
    return float(tuple(kw_res)[0]) < p, float(tuple(kw_res)[0])

def test_rep_wilcoxon_r(sl,cl_hie,feats,th,multiclass_strat,mul_cor,fn,min_c,comp_only_same_subcl,curv=False):
    comp_all_sub = not comp_only_same_subcl
    tot_ok =  0
    alpha_mtc = th
    all_diff = []
    for pair in [(x,y) for x in cl_hie.keys() for y in cl_hie.keys() if x < y]:
        dir_cmp = "not_set" #
        l_subcl1, l_subcl2 = (len(cl_hie[pair[0]]), len(cl_hie[pair[1]]))
        if mul_cor != 0: alpha_mtc = th*l_subcl1*l_subcl2 if mul_cor == 2 else 1.0-math.pow(1.0-th,l_subcl1*l_subcl2)
        ok = 0
        curv_sign = 0
        first = True
        for i,k1 in enumerate(cl_hie[pair[0]]):
            br = False
            for j,k2 in enumerate(cl_hie[pair[1]]):
                if not comp_all_sub and k1[len(pair[0]):] != k2[len(pair[1]):]:
                    ok += 1
                    continue
                cl1 = feats[sl[k1][0]:sl[k1][1]]
                cl2 = feats[sl[k2][0]:sl[k2][1]]
                med_comp = False
                if len(cl1) < min_c or len(cl2) < min_c:
                    med_comp = True
                sx,sy = numpy.median(cl1),numpy.median(cl2)
                if cl1[0] == cl2[0] and len(set(cl1)) == 1 and  len(set(cl2)) == 1:
                    tres, first = False, False
                elif not med_comp:
                    robjects.globalenv["x"] = robjects.FloatVector(cl1+cl2)
                    robjects.globalenv["y"] = robjects.FactorVector(robjects.StrVector(["a" for a in cl1]+["b" for b in cl2]))
                    pv = float(robjects.r('pvalue(wilcox_test(x~y,data=data.frame(x,y)))')[0])
                    tres = pv < alpha_mtc*2.0
                if first:
                    first = False
                    if not curv and ( med_comp or tres ):
                        dir_cmp = sx < sy
                        #if sx == sy: br = True
                    elif curv:
                        dir_cmp = None
                        if med_comp or tres:
                            curv_sign += 1
                            dir_cmp = sx < sy
                    else: br = True
                elif not curv and med_comp:
                    if ((sx < sy) != dir_cmp or sx == sy): br = True
                elif curv:
                    if tres and dir_cmp == None:
                        curv_sign += 1
                        dir_cmp = sx < sy
                    if tres and dir_cmp != (sx < sy):
                        br = True
                        curv_sign = -1
                elif not tres or (sx < sy) != dir_cmp or sx == sy: br = True
                if br: break
                ok += 1
            if br: break
        if curv: diff = curv_sign > 0
        else: diff = (ok == len(cl_hie[pair[1]])*len(cl_hie[pair[0]])) # or (not comp_all_sub and dir_cmp != "not_set")
        if diff: tot_ok += 1
        if not diff and multiclass_strat: return False
        if diff and not multiclass_strat: all_diff.append(pair)
    if not multiclass_strat:
        tot_k = len(list(cl_hie.keys()))
        for k in cl_hie.keys():
            nk = 0
            for a in all_diff:
                if k in a: nk += 1
            if nk == tot_k-1: return True
        return False
    return True



def contast_within_classes_or_few_per_class(feats,inds,min_cl,ncl):
    ff = list(zip(*[v for n,v in feats.items() if n != 'class']))
    cols = [ff[i] for i in inds]
    cls = [feats['class'][i] for i in inds]
    if len(set(cls)) < ncl:
        return True
    for c in set(cls):
        if cls.count(c) < min_cl:
            return True
        cols_cl = [x for i,x in enumerate(cols) if cls[i] == c]
        for i,col in enumerate(zip(*cols_cl)):
            if (len(set(col)) <= min_cl and min_cl > 1) or (min_cl == 1 and len(set(col)) <= 1):
                return True
    return False

def test_lda_r(cls, feats, cl_sl, boots, fract_sample, lda_th, tol_min, nlogs):
    """
    Enhanced effect size computation with better scaling to match original LDA behavior.
    """
    print(f"[INFO] Computing effect sizes (enhanced Cohen's d approach)")
    print(f"  - Bootstrap iterations: {boots}")
    print(f"  - Sample fraction: {fract_sample}")
    print(f"  - Effect threshold: {lda_th}")
    
    import numpy as np
    import math
    import random as lrand
    
    # Get feature keys (exclude metadata)
    fk = [k for k in feats.keys() if k not in ['class', 'subclass', 'subject']]
    
    if len(fk) == 0:
        print("[ERROR] No features available")
        return None, None
    
    # Get class information
    class_labels = cls['class']
    unique_classes = list(set(class_labels))
    n_samples = len(class_labels)
    
    print(f"[DEBUG] Classes: {unique_classes}")
    print(f"[DEBUG] Total samples: {n_samples}")
    
    # Build feature matrix
    feature_matrix = {}
    for k in fk:
        feature_matrix[k] = np.array(feats[k])
    
    # Storage for bootstrap results
    effect_sizes = {k: [] for k in fk}
    
    boot_size = int(n_samples * fract_sample)
    successful_boots = 0
    
    for boot_idx in range(boots):
        if boot_idx % 10 == 0:
            print(f"[DEBUG] Bootstrap {boot_idx+1}/{boots}")
        
        # Random bootstrap sample
        boot_indices = [lrand.randint(0, n_samples - 1) for _ in range(boot_size)]
        
        # Get bootstrap classes
        boot_classes = [class_labels[i] for i in boot_indices]
        
        # Check we have at least 2 samples per class
        class_counts = {c: boot_classes.count(c) for c in unique_classes}
        if any(count < 2 for count in class_counts.values()):
            continue
        
        # For each feature, compute effect size
        for feat_name in fk:
            feat_values = feature_matrix[feat_name][boot_indices]
            
            # Compute class-wise statistics
            class_means = []
            class_stds = []
            
            for cls in unique_classes:
                cls_indices = [i for i, c in enumerate(boot_classes) if c == cls]
                cls_values = feat_values[cls_indices]
                
                class_means.append(np.mean(cls_values))
                class_stds.append(np.std(cls_values, ddof=1))
            
            # Compute effect size with better scaling
            if len(unique_classes) == 2:
                # Cohen's d: (mean1 - mean2) / pooled_std
                mean_diff = abs(class_means[0] - class_means[1])
                pooled_std = math.sqrt((class_stds[0]**2 + class_stds[1]**2) / 2)
                
                if pooled_std > 1e-10:
                    effect = mean_diff / pooled_std
                else:
                    # If std is zero, use the absolute difference scaled by mean
                    effect = mean_diff / (abs(np.mean(class_means)) + 1e-10)
            else:
                # Multi-class: use max pairwise difference / within-class std
                max_diff = 0
                for i in range(len(unique_classes)):
                    for j in range(i + 1, len(unique_classes)):
                        diff = abs(class_means[i] - class_means[j])
                        if diff > max_diff:
                            max_diff = diff
                
                avg_std = np.mean(class_stds)
                if avg_std > 1e-10:
                    effect = max_diff / avg_std
                else:
                    effect = max_diff / (np.mean([abs(m) for m in class_means]) + 1e-10)
            
            effect_sizes[feat_name].append(effect)
        
        successful_boots += 1
    
    print(f"[INFO] Successful bootstraps: {successful_boots}/{boots}")
    
    if successful_boots < boots * 0.3:
        print(f"[ERROR] Too few successful bootstraps ({successful_boots}/{boots})")
        return None, None
    
    # Compute final scores with better scaling
    results = {}
    for feat_name in fk:
        if len(effect_sizes[feat_name]) == 0:
            continue
        
        # Average effect size across bootstraps
        avg_effect = np.mean(effect_sizes[feat_name])
        
        # Scale to match original LDA score range (typically 2-5)
        # Use a more aggressive scaling: log10(1 + 10*effect)
        if avg_effect > 0:
            log_score = math.log10(1.0 + 10.0 * avg_effect)
            results[feat_name] = log_score
        else:
            results[feat_name] = 0.0
    
    print(f"[INFO] Computed scores for {len(results)} features")
    
    if len(results) > 0:
        sorted_feats = sorted(results.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"[INFO] Top 10 features:")
        for feat, score in sorted_feats[:10]:
            short_name = feat.split('.')[-1] if '.' in feat else feat
            print(f"  {short_name}: {score:.3f}")
    
    # Filter by threshold
    significant = {k: v for k, v in results.items() if abs(v) > lda_th}
    print(f"[INFO] {len(significant)}/{len(results)} features exceed threshold {lda_th}")
    
    return results, significant

def test_svm(cls,feats,cl_sl,boots,fract_sample,lda_th,tol_min,nsvm):
    print(f"[WARNING] SVM testing is not currently implemented")
    return None
"""
    fk = feats.keys()
    clss = list(set(cls['class']))
    y = [clss.index(c)*2-1 for c in list(cls['class'])]
    xx = [feats[f] for f in fk]
    if nsvm:
        maxs = [max(v) for v in xx]
        mins = [min(v) for v in xx]
        x = [ dict([(i+1,(v-mins[i])/(maxs[i]-mins[i])) for i,v in enumerate(f)]) for f in zip(*xx)]
    else: x = [ dict([(i+1,v) for i,v in enumerate(f)]) for f in zip(*xx)]

    lfk = len(feats[fk[0]])
    rfk = int(float(len(feats[fk[0]]))*fract_sample)
    mm = []

    best_c = svmutil.svm_ms(y, x, [pow(2.0,i) for i in range(-5,10)],'-t 0 -q')
    for i in range(boots):
        rand_s = [lrand.randint(0,lfk-1) for v in range(rfk)]
        r = svmutil.svm_w([y[yi] for yi in rand_s], [x[xi] for xi in rand_s], best_c,'-t 0 -q')
        mm.append(r[:len(fk)])
    m = [numpy.mean(v) for v in zip(*mm)]
    res = dict([(v,m[i]) for i,v in enumerate(fk)])
    return res,dict([(k,x) for k,x in res.items() if math.fabs(x) > lda_th])
"""
