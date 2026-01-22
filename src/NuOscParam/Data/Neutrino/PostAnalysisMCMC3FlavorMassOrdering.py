import numpy as np


import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy import ndimage

import glob

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica", 
    "font.size":16})

import pandas as pd
#import seaborn as sns

from MixMat import *


class MCMC3FlavorOrderingAnalyser:
    def __init__(self,  file_name,  purity=False,  cut = 10):
        self.purity = purity
        self.cut = cut
        self.ref_values_mixing = ref_PMNS_3flavor()
        self.ref_values_mass = ref_mass_splitting_3flavor()
        self.load_pandas_data_frame(file_name)
        self.set_up_reference_values(file_name)
        self.add_variable()
    def load_pandas_data_frame(self, filename):
        if '*' in filename:
            file_list = glob.glob(filename)
            if self.purity == True:
                dfs = [pd.read_csv(file).iloc[-self.cut:] for file in file_list]
            else:
                dfs = [pd.read_csv(file) for file in file_list]
            self.df = pd.concat(dfs, ignore_index=True)
        else:
            self.df = pd.read_csv(filename)
        self.global_analysis()
    def set_up_reference_values(self, filename):
        short_name = filename.split('/')[-1]
        if 'IO' in short_name:
            self.ref_MH = 1
            self.ref_theta12 = self.ref_values_mixing.theta12_IO
            self.ref_theta13 = self.ref_values_mixing.theta13_IO
            self.ref_theta23 = self.ref_values_mixing.theta23_IO
            self.ref_m21 = self.ref_values_mass.delta_m21_IO
            self.ref_m32 = self.ref_values_mass.delta_m32_IO
            self.ref_delta_cp = self.ref_values_mixing.delta_cp_IO
            
            self.ref_theta12_n = (self.ref_values_mixing.theta12_IO-self.ref_values_mixing.theta12_IO_elow)/(self.ref_values_mixing.theta12_IO_ehigh-self.ref_values_mixing.theta12_IO_elow)
            self.ref_theta13_n = (self.ref_values_mixing.theta13_IO-self.ref_values_mixing.theta13_IO_elow)/(self.ref_values_mixing.theta13_IO_ehigh-self.ref_values_mixing.theta13_IO_elow)
            self.ref_theta23_n = (self.ref_values_mixing.theta23_IO-self.ref_values_mixing.theta23_IO_elow)/(self.ref_values_mixing.theta23_IO_ehigh-self.ref_values_mixing.theta23_IO_elow)
            self.ref_delta_cp_n = (self.ref_values_mixing.delta_cp_IO-self.ref_values_mixing.delta_cp_IO_elow)/(self.ref_values_mixing.delta_cp_IO_ehigh-self.ref_values_mixing.delta_cp_IO_elow)
            self.ref_m21_n = (self.ref_values_mass.delta_m21_IO-self.ref_values_mass.delta_m21_IO_elow)/(self.ref_values_mass.delta_m21_IO_ehigh-self.ref_values_mass.delta_m21_IO_elow)
            self.ref_m32_n = (self.ref_values_mass.delta_m32_IO-self.ref_values_mass.delta_m32_IO_elow)/(self.ref_values_mass.delta_m32_IO_ehigh-self.ref_values_mass.delta_m32_IO_elow)
        else:
            self.ref_MH = 0
            self.ref_theta12 = self.ref_values_mixing.theta12_NO
            self.ref_theta13 = self.ref_values_mixing.theta13_NO
            self.ref_theta23 = self.ref_values_mixing.theta23_NO
            self.ref_m21 = self.ref_values_mass.delta_m21_NO
            self.ref_m32 = self.ref_values_mass.delta_m32_NO
            self.ref_delta_cp = self.ref_values_mixing.delta_cp_NO
            
            self.ref_theta12_n = (self.ref_values_mixing.theta12_NO-self.ref_values_mixing.theta12_NO_elow)/(self.ref_values_mixing.theta12_NO_ehigh-self.ref_values_mixing.theta12_NO_elow)
            self.ref_theta13_n = (self.ref_values_mixing.theta13_NO-self.ref_values_mixing.theta13_NO_elow)/(self.ref_values_mixing.theta13_NO_ehigh-self.ref_values_mixing.theta13_NO_elow)
            self.ref_theta23_n = (self.ref_values_mixing.theta23_NO-self.ref_values_mixing.theta23_NO_elow)/(self.ref_values_mixing.theta23_NO_ehigh-self.ref_values_mixing.theta23_NO_elow)
            self.ref_delta_cp_n = (self.ref_values_mixing.delta_cp_NO-self.ref_values_mixing.delta_cp_NO_elow)/(self.ref_values_mixing.delta_cp_NO_ehigh-self.ref_values_mixing.delta_cp_NO_elow)
            self.ref_m21_n = (self.ref_values_mass.delta_m21_NO-self.ref_values_mass.delta_m21_NO_elow)/(self.ref_values_mass.delta_m21_NO_ehigh-self.ref_values_mass.delta_m21_NO_elow)
            self.ref_m32_n = (self.ref_values_mass.delta_m32_NO-self.ref_values_mass.delta_m32_NO_elow)/(self.ref_values_mass.delta_m32_NO_ehigh-self.ref_values_mass.delta_m32_NO_elow)
            
    def normalize_variable(self, column_name, x_min_0, x_max_0, x_min_1, x_max_1):
        self.df[column_name + '_norm'] = self.df.apply(
            lambda row: (row[column_name] - x_min_0) / (x_max_0 - x_min_0)
                        if row['MH'] == 0
                        else (row[column_name] - x_min_1) / (x_max_1 - x_min_1),
            axis=1
        )
    def add_variable(self):
        self.df['m32'] = self.df.apply(lambda row: row.m3 - row.m2, axis=1)
        self.df['m21'] = self.df.apply(lambda row: row.m2 - row.m1, axis=1)
        self.normalize_variable("theta_12",  self.ref_values_mixing.theta12_NO_elow, self.ref_values_mixing.theta12_NO_ehigh,self.ref_values_mixing.theta12_IO_elow, self.ref_values_mixing.theta12_IO_ehigh )
        self.normalize_variable("theta_23",  self.ref_values_mixing.theta23_NO_elow, self.ref_values_mixing.theta23_NO_ehigh,self.ref_values_mixing.theta23_IO_elow, self.ref_values_mixing.theta23_IO_ehigh )
        self.normalize_variable("theta_13",  self.ref_values_mixing.theta13_NO_elow, self.ref_values_mixing.theta13_NO_ehigh,self.ref_values_mixing.theta13_IO_elow, self.ref_values_mixing.theta13_IO_ehigh )
        self.normalize_variable("delta_cp",  self.ref_values_mixing.delta_cp_NO_elow, self.ref_values_mixing.delta_cp_NO_ehigh,self.ref_values_mixing.delta_cp_IO_elow, self.ref_values_mixing.delta_cp_IO_ehigh )
        self.normalize_variable("m21", self.ref_values_mass.delta_m21_NO_elow, self.ref_values_mass.delta_m21_NO_ehigh, self.ref_values_mass.delta_m21_IO_elow, self.ref_values_mass.delta_m21_IO_ehigh )
        self.normalize_variable("m32", self.ref_values_mass.delta_m32_NO_elow, self.ref_values_mass.delta_m32_NO_ehigh, self.ref_values_mass.delta_m32_IO_elow, self.ref_values_mass.delta_m32_IO_ehigh )

    


    def plot_physics(self):
        variable_name = ['theta_12_norm', 'theta_23_norm', 'theta_13_norm', 'delta_cp_norm', 'm21_norm',  'm32_norm']
        variable_label = ['$\\theta_{12}$', '$\\theta_{23}$', '$\\theta_{13}$', '$\\delta_{CP}$', '$m_{21}$',  '$m_{32}$']
        variable_ref = [self.ref_theta12_n, self.ref_theta23_n, self.ref_theta13_n, self.ref_delta_cp_n, self.ref_m21_n,  self.ref_m32_n]
        if self.ref_MH == 0:
            variable_ref_color = 'b'
        else:
            variable_ref_color = 'r'
        id_max = self.df["LL"].idxmax()
        if self.df.iloc[id_max]['MH'] == 0:
            bf_fit_color = 'b'
        else:
            bf_fit_color = 'r'
        
        ndim = len(variable_name)
        x_bin = np.linspace(0, 1, 51)
        y_bin = np.linspace(0, 1, 51)
        sigma_filter = 1.
        X_mesh, Y_mesh = np.meshgrid((x_bin[:-1]+x_bin[1:])*0.5, (y_bin[:-1]+y_bin[:-1])*0.5)
        
        fig, axes = plt.subplots(ndim, ndim, figsize=(12, 12))
        for ax in axes.flatten():
                ax.yaxis.set_ticks([])
                ax.xaxis.set_ticks([])
            
        
        
        for i,  v in enumerate(variable_name) :
            print(v)
            hist_NO, bin_edges = np.histogram(self.df[self.df['MH'] == 0][v], bins=x_bin)
            hist_IO, bin_edges = np.histogram(self.df[self.df['MH'] == 1][v], bins=x_bin)
            
            hist_IO = ndimage.gaussian_filter1d(hist_IO, sigma=sigma_filter)
            hist_NO = ndimage.gaussian_filter1d(hist_NO, sigma=sigma_filter)
                
            maxNO = np.max(hist_NO)
            maxIO = np.max(hist_IO)
            max = np.max([maxIO, maxNO])
            axes[i][i].bar(x_bin[:-1], hist_NO, width=(x_bin[1] - x_bin[0]), align='edge', color='b', alpha=0.4)
            axes[i][i].bar(x_bin[:-1], hist_IO, width=(x_bin[1] - x_bin[0]), align='edge',color='r', alpha=0.4)
            axes[i][i].plot([variable_ref[i],variable_ref[i]], [0, max], color=variable_ref_color)
            axes[i][i].plot([self.df.iloc[id_max][v],self.df.iloc[id_max][v]], [0, max], ':'+bf_fit_color)
            axes[i][i].set_xlim(0, 1)
            for j in range(ndim-i-1):
                hist_NO, xbin_edges,  ybin_edges = np.histogram2d(self.df[self.df['MH'] == 0][v], self.df[self.df['MH'] == 0][variable_name[j+i+1]] , bins=[x_bin, y_bin])
                hist_IO, xbin_edges,  ybin_edges = np.histogram2d(self.df[self.df['MH'] == 1][v], self.df[self.df['MH'] == 1][variable_name[j+i+1]] , bins=[x_bin, y_bin])
                hist_NO = ndimage.gaussian_filter(hist_NO, sigma=sigma_filter)
                hist_IO = ndimage.gaussian_filter(hist_IO, sigma=sigma_filter)
                #levels_IO = np.logspace(np.log(np.max( hist_IO)/1000.), np.log(np.max( hist_IO)), 5)
                #levels_NO = np.logspace(np.log(np.max( hist_NO)/1000.) ,np.log(np.max( hist_NO)), 5)
                levels_IO = np.linspace(0, np.max([np.max( hist_NO), np.max( hist_IO)]), 30)
                levels_NO = np.linspace(0 ,np.max([np.max( hist_NO), np.max( hist_IO)]), 30)
                levels_IO = np.linspace(0, np.max( hist_IO), 30)
                levels_NO = np.linspace(0 ,np.max( hist_NO), 30)
                #axes[j+i+1][i].contourf(X_mesh, Y_mesh, hist_IO.T,   cmap=colormaps['Reds'], levels=levels_IO[:],alpha=0.7)# linewidths=1, linestyles='solid' )
                axes[j+i+1][i].contourf(X_mesh, Y_mesh, hist_NO.T,  cmap=colormaps['Blues'], levels=levels_NO[:], alpha=0.7)#linewidths=1, linestyles='solid' )
                axes[j+i+1][i].plot(variable_ref[i], variable_ref[j+i+1], '*'+variable_ref_color)
                axes[j+i+1][i].plot(self.df.iloc[id_max][v], self.df.iloc[id_max][variable_name[j+i+1]], 'o'+bf_fit_color)
                
                axes[i][j+i+1].contourf( Y_mesh,X_mesh, hist_IO.T,   cmap=colormaps['Reds'], levels=levels_IO[:],alpha=0.7)# linewidths=1, linestyles='solid' )
                axes[i][j+i+1].plot(variable_ref[j+i+1],variable_ref[i],  '*'+variable_ref_color)
                axes[i][j+i+1].plot(self.df.iloc[id_max][variable_name[j+i+1]],self.df.iloc[id_max][v],  'o'+bf_fit_color)
            if i < ndim-1:
                axes[i+1][0].set_ylabel(variable_label[i+1])
                axes[i+1][0].yaxis.set_ticks([0.25, 0.5, 0.75])
            axes[ndim-1][i].set_xlabel(variable_label[i])
            axes[ndim-1][ndim-1].set_xlabel(variable_label[-1])
            axes[ndim-1][i].xaxis.set_ticks([0,  0.5, 1])
            axes[ndim-1][ndim-1].xaxis.set_ticks([0,  0.5, 1])
        plt.tight_layout()

        plt.subplots_adjust(wspace=0, hspace=0)
        #plt.show()

    def plot_convergence(self):
        fig, ax = plt.subplots(2, 1, figsize=(12, 12))
        ax[0].set_title("convergence")
        ax[0].set_xlabel("MCMC step")
        ax[0].set_ylabel("LL")
        ax[0].plot(self.df['Unnamed: 0'], self.df['LL'])
        ax[1].hist(self.df['LL'], bins=50)
        ax[1].set_xlabel("LL")
    def replay_LL(self,  n_mcmc):
        nentry = len(self.df.index)
        LL = self.df["LL"].to_numpy()
        i_list = []

        random_event = np.random.randint(nentry, size=n_mcmc)
        rand_test = np.random.uniform(0., 1., n_mcmc)
        #print(LL)
        #print(random_event)
        LL_0 = LL[random_event[0]]
        i_0 = random_event[0]
        for i,  r  in enumerate( rand_test):
            if r < np.exp(LL[random_event[i]]-LL_0):
                LL_0 = LL[random_event[i]]
                i_0 = random_event[i]
            i_list.append(i_0)
            
        self.plot_physics_list(i_list)
            
    def plot_physics_list(self, list):
        variable_name = ['theta_12_norm', 'theta_23_norm', 'theta_13_norm', 'delta_cp_norm', 'm21_norm',  'm32_norm']
        variable_label = ['$\\theta_{12}$', '$\\theta_{23}$', '$\\theta_{13}$', '$\\delta_{CP}$', '$m_{21}$',  '$m_{32}$']
        variable_ref = [self.ref_theta12_n, self.ref_theta23_n, self.ref_theta13_n, self.ref_delta_cp_n, self.ref_m21_n,  self.ref_m32_n]
        if self.ref_MH == 0:
            variable_ref_color = 'b'
        else:
            variable_ref_color = 'r'
        ndim = len(variable_name)
        x_bin = np.linspace(0, 1, 51)
        y_bin = np.linspace(0, 1, 51)
        sigma_filter = 1.5
        X_mesh, Y_mesh = np.meshgrid((x_bin[:-1]+x_bin[1:])*0.5, (y_bin[:-1]+y_bin[:-1])*0.5)

        id_max = self.df["LL"].idxmax()
        if self.df.iloc[id_max]['MH'] == 0:
            bf_fit_color = 'b'
        else:
            bf_fit_color = 'r'

        
        fig, axes = plt.subplots(ndim, ndim, figsize=(12, 12))
        for ax in axes.flatten():
                ax.yaxis.set_ticks([])
                ax.xaxis.set_ticks([])
            
        
        ldf = self.df.iloc[list]
        print(ldf)
        for i,  v in enumerate(variable_name) :
            print(v)
            hist_NO, bin_edges = np.histogram(ldf[ldf['MH'] == 0][v], bins=x_bin)
            hist_IO, bin_edges = np.histogram(ldf[ldf['MH'] == 1][v], bins=x_bin)
            
            hist_IO = ndimage.gaussian_filter1d(hist_IO, sigma=sigma_filter)
            hist_NO = ndimage.gaussian_filter1d(hist_NO, sigma=sigma_filter)
                
            maxNO = np.max(hist_NO)
            maxIO = np.max(hist_IO)
            max = np.max([maxIO, maxNO])
            axes[i][i].bar(x_bin[:-1], hist_NO, width=(x_bin[1] - x_bin[0]), align='edge', color='b', alpha=0.4)
            axes[i][i].bar(x_bin[:-1], hist_IO, width=(x_bin[1] - x_bin[0]), align='edge',color='r', alpha=0.4)
            axes[i][i].plot([variable_ref[i],variable_ref[i]], [0, max], color=variable_ref_color)
            axes[i][i].plot([self.df.iloc[id_max][v],self.df.iloc[id_max][v]], [0, max], ':'+bf_fit_color)
            axes[i][i].set_xlim(0, 1)
            for j in range(ndim-i-1):
                hist_NO, xbin_edges,  ybin_edges = np.histogram2d(ldf[ldf['MH'] == 0][v], ldf[ldf['MH'] == 0][variable_name[j+i+1]] , bins=[x_bin, y_bin])
                hist_IO, xbin_edges,  ybin_edges = np.histogram2d(ldf[ldf['MH'] == 1][v], ldf[ldf['MH'] == 1][variable_name[j+i+1]] , bins=[x_bin, y_bin])
                hist_NO = ndimage.gaussian_filter(hist_NO, sigma=sigma_filter)
                hist_IO = ndimage.gaussian_filter(hist_IO, sigma=sigma_filter)
                #levels_IO = np.logspace(np.log(np.max( hist_IO)/1000.), np.log(np.max( hist_IO)), 5)
                #levels_NO = np.logspace(np.log(np.max( hist_NO)/1000.) ,np.log(np.max( hist_NO)), 5)
                levels_IO = np.linspace(0, np.max([np.max( hist_NO), np.max( hist_IO)]), 30)
                levels_NO = np.linspace(0 ,np.max([np.max( hist_NO), np.max( hist_IO)]), 30)
                levels_IO = np.linspace(0, np.max( hist_IO), 30)
                levels_NO = np.linspace(0 ,np.max( hist_NO), 30)
                #axes[j+i+1][i].contourf(X_mesh, Y_mesh, hist_IO.T,   cmap=colormaps['Reds'], levels=levels_IO[:],alpha=0.7)# linewidths=1, linestyles='solid' )
                axes[j+i+1][i].contourf(X_mesh, Y_mesh, hist_NO.T,  cmap=colormaps['Blues'], levels=levels_NO[:], alpha=0.7)#linewidths=1, linestyles='solid' )
                axes[j+i+1][i].plot(variable_ref[i], variable_ref[j+i+1], '*'+variable_ref_color)
                axes[j+i+1][i].plot(self.df.iloc[id_max][v], self.df.iloc[id_max][variable_name[j+i+1]], 'o'+bf_fit_color)
                
                axes[i][j+i+1].contourf( Y_mesh, X_mesh,hist_IO.T,   cmap=colormaps['Reds'], levels=levels_IO[:],alpha=0.7)# linewidths=1, linestyles='solid' )
                axes[i][j+i+1].plot( variable_ref[j+i+1],variable_ref[i], '*'+variable_ref_color)
                axes[i][j+i+1].plot( self.df.iloc[id_max][variable_name[j+i+1]],self.df.iloc[id_max][v], 'o'+bf_fit_color)
            if i < ndim-1:
                axes[i+1][0].set_ylabel(variable_label[i+1])
                axes[i+1][0].yaxis.set_ticks([0.25, 0.5, 0.75])
            axes[ndim-1][i].set_xlabel(variable_label[i])
            axes[ndim-1][ndim-1].set_xlabel(variable_label[-1])
            axes[ndim-1][i].xaxis.set_ticks([0,  0.5, 1])
            axes[ndim-1][ndim-1].xaxis.set_ticks([0,  0.5, 1])

        plt.tight_layout()

        plt.subplots_adjust(wspace=0, hspace=0)
        
        
        
        


    def global_analysis(self):
        LL = self.df["LL"].to_numpy()
        nentry = len(self.df.index)

        number_of_change = 1
        for i,  l in enumerate(LL):
            if i > 0:
                if l != LL[i-1]:
                    number_of_change = number_of_change+1
        print("acceptation rate = "+str(float(number_of_change)/float(nentry)*100.))
        
        
        
