import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.stats import expon
import pickle

class ColAvoidIntent:
    def __init__(self, data_dir):
        self.dir = data_dir
        self.M = 0
        self.B = 0
        self.filenames = os.listdir(data_dir+'/test')
        self.ped_frames = {}
        self.gt_cls = {}
        self.pred_cls = {}
        self.ignore_ped = {}
        rv = expon()
        self.weights = rv.pdf(np.linspace(0.2, 0.9, num=12))
        self.threshhold = 4
        self.get_results()

    def filter_traj(self, dataset):
        arr = []
        self.ignore_ped = {}
        gt = self.get_data(self.dir+'/test_private/'+dataset, obs=True)
        self.gt_cls[dataset] = []
        self.pred_cls[dataset] = {}
        
        for ped in gt:
            m, b = self.fit_line(gt[ped]['x'][:10], gt[ped]['y'][:10])
            
            dis1 = self.get_traj_classifications({'x': gt[ped]['x'][10:], 'y':gt[ped]['y'][10:]}, value=True)
            dis2 = self.get_traj_classifications({'x': gt[ped]['x'][:10], 'y':gt[ped]['y'][:10]}, value=True)
            if abs(dis1)<self.threshhold or abs(dis2)>self.threshhold:
                self.ignore_ped[ped] = 1
            
        with open(dataset.split('.')[0],'wb') as f:
            pickle.dump(self.ignore_ped,f)
    


    def get_results(self):
        models = [mod for mod in os.listdir(self.dir+'/test_pred') if mod[0] != '.']
        error_thresh = {}
        for th in range(5,6):
            self.threshhold = th
            for dataset in self.filenames:
               
                
                self.ped_frames = {}
                self.set_pred_frames(self.dir+'/test_private/'+dataset, self.dir+'/test_pred/'+models[0]+'/'+dataset)
                self.filter_traj(dataset)
               
                obs = self.get_data(self.dir+'/test/'+dataset, obs=True)
                
                gt = self.get_data(self.dir+'/test_private/'+dataset)
                
                self.gt_cls[dataset] = []
                self.pred_cls[dataset] = {}
                preds = {}
                
                for model in models:
                    preds[model] = self.get_data(self.dir+'/test_pred/'+model+'/'+dataset)
                    
                for ped in obs:
                    if ped in self.ignore_ped:
                        continue
                    self.M, self.B = self.fit_line(obs[ped]['x'], obs[ped]['y'])
                    self.gt_cls[dataset].append(self.get_traj_classifications(gt[ped], value=False))
               

                for model in preds:
                    pred = preds[model]
                    self.pred_cls[dataset][model] = []
                    for ped in obs:
                        if ped in self.ignore_ped:
                            continue
                        self.pred_cls[dataset][model].append(self.get_traj_classifications(pred[ped], value=False))

            p_cls = {}
            for model in models:
                p_cls[model] = {}
                for dataset in self.filenames:
                    p_cls[model][dataset] = self.pred_cls[dataset][model]
            
            self.pred_cls = p_cls
           
            me = []
            for model in models:
                print(model)
                error = 0
                dat = 0
                for dataset in self.filenames:
                    dat += len(self.gt_cls[dataset])
                    for v1,v2 in zip(self.gt_cls[dataset], self.pred_cls[model][dataset]):
                        error += abs(v1-v2)
                if model in error_thresh:
                    error_thresh[model].append(1-error/dat)
                else:
                    error_thresh[model] = [1-error/dat]
                print("Accuracy:", 1-error/dat)

    def get_traj_classifications(self, traj, value=False):
        traj_cls = []
        dis = []
        weights = self.weights[:len(traj['x'])]
        for x1,y1 in zip(traj['x'], traj['y']):
            dis.append(self.distance_from_line(self.M,self.B,x1,y1))
        cum_val = 0
        for wt, d in zip(weights, dis):
            cum_val += wt*d
        if not value:   
            return int(cum_val>0)
        else:
            return cum_val

    def point_left_or_right(self,x1,y1):
        l_or_r = self.distance_from_line(self.M,self.B,x1,y1) > 0
        return int(l_or_r)
    
    def distance_from_line(self, w, c, x1, y1):
        return (w*x1 - y1 + c) / math.sqrt(w*w + 1)
    
    def set_pred_frames(self, file1, file2):
        with open(file1) as gt:
            for line in gt:
                line = line.replace('"','')
                line = line[1:-1]
                [tp, line] = line.split(':', 1)
                if tp == 'scene':
                    continue
                line = line[2:-2]
                data = line.split(',')
                f, p, x, y = int(data[0].split(':')[1][1:]), int(data[1].split(':')[1][1:]), float(data[2].split(':')[1][1:]), float(data[3].split(':')[1][1:])
                if p in self.ped_frames:
                    self.ped_frames[p]['ub'] = f
                else:
                    self.ped_frames[p] = {'lb':f, 'ub':-1}

    def get_data(self, file_path, obs=False):
        paths = {}
        with open(file_path) as fl:
            for line in fl:
                line = line.replace('"','')
                line = line[1:-1]
                [tp, line] = line.split(':', 1)
                if tp == 'scene':
                    continue
                line = line[2:-2]
                data = line.split(',')
                f, p, x, y = int(data[0].split(':')[1][1:]), int(data[1].split(':')[1][1:]), float(data[2].split(':')[1][1:]), float(data[3].split(':')[1][1:])

                if p not in paths:
                    paths[p] = {'x':[], 'y':[]}
                if obs:
                    paths[p]['x'].append(x)
                    paths[p]['y'].append(y)
                elif not obs and f >= self.ped_frames[p]['lb'] and f<= self.ped_frames[p]['ub'] and len(paths[p]['x'])<12:
                    paths[p]['x'].append(x)
                    paths[p]['y'].append(y)
        return paths



    def fit_line(self, points_X, points_Y):
        X = np.array(points_X)
        Y = np.array(points_Y)
        m, b = np.polyfit(X,Y,1)
        return m,b

def main():
    met = ColAvoidIntent('../code/trajnetplusplusbaelines-1/DATA_BLOCK/orcadata/')

if __name__ == '__main__':
    main()