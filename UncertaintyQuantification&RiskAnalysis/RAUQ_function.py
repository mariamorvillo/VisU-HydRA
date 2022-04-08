import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LinearLocator, FuncFormatter, FormatStrFormatter
from copy import copy
import os
import pickle

path = 'figures/'
if not os.path.exists(path):
    os.makedirs(path)
    
class plotinfo:
    def __init__(self, n_realization, Kg, Lx, Ly, block_x, block_y, lambda_x, lambda_y, 
                 source_xl, source_xu, source_yl, source_yu, 
                 target_xl, target_xu, target_yl, target_yu,
                 mcl, observation_wells, tstep, dt):
        
        self.n_realization = n_realization
        self.Kg = Kg
        self.Lx = Lx
        self.Ly = Ly
        self.block_x = block_x
        self.block_y = block_y
        self.lambda_x = lambda_x 
        self.lambda_y = lambda_y
        self.source_xl = source_xl 
        self.source_xu = source_xu
        self.source_yl = source_yl 
        self.source_yu = source_yu
        self.target_xl = target_xl
        self.target_xu = target_xu
        self.target_yl = target_yl
        self.target_yu = target_yu
        self.mcl = mcl
        self.observation_wells = np.asarray(observation_wells)
        self.tstep = tstep
        self.nt = len(tstep)
        self.dt = dt
        
        self.kfields = np.load('Kfileds_Hydrogen.npy')
        self.kmax = np.ceil(np.max(self.kfields))
        self.kmin = np.floor(np.min(self.kfields))
        

    def logkfield(self, filename, real_n):
        kfield = self.kfields[real_n]
        fig, ax = plt.subplots(figsize=(7,5))
        img = ax.imshow(kfield, cmap='jet', extent=[0,self.Lx/self.lambda_x,self.Ly/self.lambda_y,0], 
                        vmin=self.kmin, vmax=self.kmax)
        rectangle1 = plt.Rectangle((self.source_xl/self.lambda_x,self.source_yl/self.lambda_y), 
                                  (self.source_xu-self.source_xl)/self.lambda_x, 
                                  (self.source_yu-self.source_yl)/self.lambda_y, 
                                  fc='k', fill=None, linewidth=1.5)
        plt.gca().add_patch(rectangle1)
        plt.text(self.source_xl/self.lambda_x, (self.source_yl-self.lambda_y)/self.lambda_y, r'source', fontsize=15, color='k')
        
        rectangle2 = plt.Rectangle((self.target_xl/self.lambda_x,self.target_yl/self.lambda_y), 
                                  (self.target_xu-self.target_xl)/self.lambda_x, 
                                  (self.target_yu-self.target_yl)/self.lambda_y, 
                                  fc='k', fill=None, linewidth=1.5)
        plt.gca().add_patch(rectangle2)       
        plt.text(self.target_xl/self.lambda_x, (self.target_yl-self.lambda_y)/self.lambda_y, r'target', fontsize=15, color='k')

        ax.scatter(self.observation_wells.T[0]/self.lambda_x, self.observation_wells.T[1]/self.lambda_y, 
                   color='k', marker='^' ,s=50, label='observation well')
        
        cbar = ax.figure.colorbar(img, ax=ax, fraction=0.041, pad=0.04)
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.set_ylabel(r'$log~K$', fontsize=25, fontname='Arial', labelpad=10)
        ax.xaxis.set_major_locator(LinearLocator(5))
        ax.xaxis.set_minor_locator(LinearLocator(21))
        ax.yaxis.set_major_locator(LinearLocator(5))
        ax.yaxis.set_minor_locator(LinearLocator(21))

        plt.xlim(0,self.Lx/self.lambda_x)
        plt.ylim(0,self.Ly/self.lambda_y)
        plt.xticks(fontsize=15, fontname='Arial')
        plt.yticks(fontsize=15, fontname='Arial')
        plt.xlabel(r'$x~/~\lambda_x$', fontsize=25, fontname='Arial', labelpad=5)
        plt.ylabel(r'$y~/~\lambda_y$', fontsize=25, fontname='Arial', labelpad=5)

        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'figures/{filename}.png',dpi=100, bbox_inches='tight')
        plt.show()

    def cfield_postprocessing(self):
        cell_n = (self.Ly, self.Lx)
        block_s = (self.block_y, self.block_x)
        name_datafiles = [f'output/snap-{i}-{j}.csv' for i in np.arange(self.n_realization) for j in self.tstep]
        data = pd.read_csv(name_datafiles[0])
        particle_n = data.count()[0]
        
        for real in range(self.n_realization):
            print(f'realization no. {real}')
            current_percent = 0
            datafiles = name_datafiles[real*self.nt:real*self.nt+self.nt]
            field_c = np.zeros((len(datafiles), cell_n[0], cell_n[1]))
            data_arrays = np.zeros((len(datafiles), 2, particle_n))

            for i in range(len(datafiles)):
                coordinates = np.floor(pd.read_csv(datafiles[i]).to_numpy('float').T[1:3]).astype(int)
                for j in range(particle_n):
                    field_c[i, coordinates[1,j]-1, coordinates[0,j]-1] += 1/particle_n

                if i/(len(datafiles)-1)*100 >= current_percent:
                    print(f'processing {current_percent}%')
                    current_percent += 50

            np.save(f'data_output/cfields/cfield_{real}', field_c)
        
    def referencepoints_postprocessing(self):
        field_c = np.load(f'data_output/cfields/cfield_0.npy')
        c0 = field_c[0,:,:-2].max()

        for real in range(self.n_realization):
            field_c = np.load(f'data_output/cfields/cfield_{real}.npy')

            maxconc_data = {'tstep': [], 'x_coord': [], 'y_coord': [], 'conc': []}
            for j in range(self.nt):
                if field_c[j,:,:-self.lambda_x].max() <= c0*self.mcl:
                    break
                else:
                    coordinates = np.where( field_c[j,:,:-self.lambda_x]==field_c[j,:,:-self.lambda_x].max())        
                    for k in range(len(coordinates[0])):
                        maxconc_data['tstep'].append(self.dt*j)
                        maxconc_data['y_coord'].append(coordinates[0][k])
                        maxconc_data['x_coord'].append(coordinates[1][k])
                        maxconc_data['conc'].append(field_c[j][maxconc_data['y_coord'][-1],maxconc_data['x_coord'][-1]])

            maxconc_data['tstep'] = np.array(maxconc_data['tstep'])
            maxconc_data['y_coord'] = np.array(maxconc_data['y_coord'])
            maxconc_data['x_coord'] = np.array(maxconc_data['x_coord'])
            maxconc_data['conc'] = np.array(maxconc_data['conc'])
            pickle.dump(maxconc_data, open(f'data_output/referencepoints/maxconc_{real}.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)    

            edge_data = {'tstep': [], 'x_coord': [], 'y_coord': []}
            for j in range(self.nt):
                edge_x = np.amax(np.argwhere(field_c[j])[:,1])
                if edge_x >= self.Lx-self.lambda_x:
                    break
                edge_y = np.argmax(field_c[j][:,edge_x])
                edge_data['tstep'].append(self.dt*j)
                edge_data['x_coord'].append(edge_x)
                edge_data['y_coord'].append(edge_y)

            edge_data['tstep'] = np.array(edge_data['tstep'])
            edge_data['x_coord'] = np.array(edge_data['x_coord'])    
            edge_data['y_coord'] = np.array(edge_data['y_coord'])
            pickle.dump(edge_data, open(f'data_output/referencepoints/edge_{real}.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        
    def cfield_ensemble_postprocessing(self):
        cfield_all = []
        for i in range(self.n_realization):
            cfield_all.append(np.load(f'data_output/cfields/cfield_{i}.npy'))
        cfield_all = np.asarray(cfield_all)
        np.save(f'data_output/cfields/cfield_all.npy', cfield_all)
        cfield_ave = cfield_all.mean(axis=0)
        np.save(f'data_output/cfields/cfield_ensemble.npy', cfield_ave)
        cfield_var = cfield_all.var(axis=0)
        np.save(f'data_output/cfields/cfield_ensemble_v.npy', cfield_var)
        
    def cfield(self, filename, real_n, time_index, plume_edge, max_conc):
        field_c = np.load(f'data_output/cfields/cfield_{real_n}.npy')
        risk_var = np.load('data_output/cfields/cfield_ensemble_v.npy', )
        if not real_n == 'ensemble':
            kfield = self.kfields[real_n]
            alpha_v = 0.7
            maxconc = pickle.load(open(f'data_output/referencepoints/maxconc_{real_n}.pkl', 'rb'))
            edge = pickle.load(open(f'data_output/referencepoints/edge_{real_n}.pkl', 'rb'))
            ylabel = r'$c$'
        else:
            alpha_v = 1
            ylabel = r'$\left< c \right>$'
            
        c0 = field_c[0].max()

        for i in time_index:
            fig, ax = plt.subplots(figsize=(7,5))
            cmap = copy(plt.get_cmap('Greens'))
            def fmt(x, pos):
                a, b = '{:.0e}'.format(x).split('e')
                b = int(b)
                return r'${} \times 10^{{{}}}$'.format(a, b)
            if not real_n == 'ensemble':
                img = ax.imshow(kfield, cmap='jet', extent=[0,self.Lx/self.lambda_x,self.Ly/self.lambda_y,0], 
                                vmin=self.kmin, vmax=self.kmax)
            img = ax.imshow(field_c[i,:,:-1]/c0, cmap=cmap, 
                            extent=[0,self.Lx/self.lambda_x,self.Ly/self.lambda_y,0], vmin=0, vmax=1e0, alpha=alpha_v)
            cbar = ax.figure.colorbar(img, ax=ax, fraction=0.041, pad=0.04, format=FuncFormatter(fmt))
            cbar.ax.tick_params(labelsize=14)
            cbar.ax.set_ylabel(ylabel, fontsize=25, fontname='Arial', labelpad=10)
            cbar.solids.set_edgecolor("face")
            ax.scatter(self.observation_wells.T[0]/self.lambda_x, self.observation_wells.T[1]/self.lambda_y, 
                       color='k', marker='^' ,s=50)            
            
            if not real_n == 'ensemble':
                if plume_edge:
                    edge_index = edge['tstep']//self.dt == i
                    edge_indexes = edge['tstep']//self.dt <= i

                    ax.scatter(edge['x_coord'][edge_index]/self.lambda_x, edge['y_coord'][edge_index]/self.lambda_y,
                               s=80, c='b', marker='X', linewidths=.01, alpha=0.8, label='plume edge')
                    ax.plot(edge['x_coord'][edge_indexes]/self.lambda_x, edge['y_coord'][edge_indexes]/self.lambda_y,
                            c='b', linestyle=':', alpha=0.8)
                    plt.legend(loc=2, fontsize=12)

                if max_conc:
                    maxconc_index = maxconc['tstep']//self.dt == i
                    maxconc_indexes = maxconc['tstep']//self.dt <= i

                    ax.scatter(maxconc['x_coord'][maxconc_index]/self.lambda_x, maxconc['y_coord'][maxconc_index]/self.lambda_y,
                               s=80, c='r', marker='X', linewidths=.01, alpha=0.8, label='max C')
                    ax.scatter(maxconc['x_coord'][maxconc_indexes]/self.lambda_x, maxconc['y_coord'][maxconc_indexes]/self.lambda_y,
                               s=30, c='r', marker='X', linewidths=.01, alpha=0.5)
                    plt.legend(loc=2, fontsize=12)

            ax.xaxis.set_major_locator(LinearLocator(5))
            ax.xaxis.set_minor_locator(LinearLocator(21))
            ax.yaxis.set_major_locator(LinearLocator(5))
            ax.yaxis.set_minor_locator(LinearLocator(21))

            rectangle2 = plt.Rectangle((self.target_xl/self.lambda_x,self.target_yl/self.lambda_y), 
                                      (self.target_xu-self.target_xl)/self.lambda_x, 
                                      (self.target_yu-self.target_yl)/self.lambda_y, 
                                      fc='k', fill=None, linewidth=1.5)
            plt.gca().add_patch(rectangle2)       

            plt.xlim(0,self.Lx/self.lambda_x)
            plt.ylim(0,self.Ly/self.lambda_y)
            plt.xticks(fontsize=15, fontname='Arial')
            plt.yticks(fontsize=15, fontname='Arial')
            plt.xlabel(r'$x~/~\lambda_x$', fontsize=25, fontname='Arial', labelpad=5)
            plt.ylabel(r'$y~/~\lambda_y$', fontsize=25, fontname='Arial', labelpad=5)

            plt.text(5/self.lambda_x, 5/self.lambda_y, f'$t={i*self.dt}$', fontsize=15, color='k')
            plt.tight_layout()
            plt.savefig(f'figures/{filename}_{i}.png',dpi=200, bbox_inches='tight')
            
            if real_n == 'ensemble':
                fig, ax = plt.subplots(figsize=(7,5))
                cmap = copy(plt.get_cmap('Purples'))
                ylabel = r'$\sigma^2_{c}$'
                def fmt(x, pos):
                    a, b = '{:.0e}'.format(x).split('e')
                    b = int(b)
                    return r'${} \times 10^{{{}}}$'.format(a, b)
                img = ax.imshow(risk_var[i,:,:-1], cmap=cmap, 
                                extent=[0,self.Lx/self.lambda_x,self.Ly/self.lambda_y,0], 
                                vmin=0, vmax=np.max(risk_var[i,:,:-self.lambda_x]), alpha=alpha_v)
                cbar = ax.figure.colorbar(img, ax=ax, fraction=0.041, pad=0.04, format=FuncFormatter(fmt))
                cbar.ax.tick_params(labelsize=14)
                cbar.ax.set_ylabel(ylabel, fontsize=25, fontname='Arial', labelpad=10)
                cbar.solids.set_edgecolor("face")
                ax.scatter(self.observation_wells.T[0]/self.lambda_x, self.observation_wells.T[1]/self.lambda_y, 
                           color='k', marker='^' ,s=50)

                ax.xaxis.set_major_locator(LinearLocator(5))
                ax.xaxis.set_minor_locator(LinearLocator(21))
                ax.yaxis.set_major_locator(LinearLocator(5))
                ax.yaxis.set_minor_locator(LinearLocator(21))

                rectangle2 = plt.Rectangle((self.target_xl/self.lambda_x,self.target_yl/self.lambda_y), 
                                          (self.target_xu-self.target_xl)/self.lambda_x, 
                                          (self.target_yu-self.target_yl)/self.lambda_y, 
                                          fc='k', fill=None, linewidth=1.5)
                plt.gca().add_patch(rectangle2)       

                plt.xlim(0,self.Lx/self.lambda_x)
                plt.ylim(0,self.Ly/self.lambda_y)
                plt.xticks(fontsize=15, fontname='Arial')
                plt.yticks(fontsize=15, fontname='Arial')
                plt.xlabel(r'$x~/~\lambda_x$', fontsize=25, fontname='Arial', labelpad=5)
                plt.ylabel(r'$y~/~\lambda_y$', fontsize=25, fontname='Arial', labelpad=5)

                plt.text(5/self.lambda_x, 5/self.lambda_y, f'$t={i*self.dt}$', fontsize=15, color='k')
                plt.tight_layout()
                plt.savefig(f'figures/{filename}_v_{i}.png',dpi=200, bbox_inches='tight')            
                ylabel = r'$\left< c \right>$'
            

    def rrfield_postprocessing(self):
        cfield_all = np.load('data_output/cfields/cfield_all.npy')
        reliability_field = np.zeros(cfield_all.shape)
        resilience_field = np.zeros((self.n_realization, self.Ly, self.Lx))
        for real_n in range(self.n_realization):
            field_c = cfield_all[real_n]
            c0 = field_c[0].max()
            reliability_field[real_n] = np.where(field_c >= (self.mcl*c0), 1, 0)
            resilience_field[real_n] = np.sum(reliability_field[real_n], axis=0)*self.dt
        risk_ensemble = reliability_field.mean(axis=0)
        risk_var = reliability_field.var(axis=0)
        np.save('data_output/reliability_field', reliability_field)
        np.save('data_output/risk_ensemble', risk_ensemble)
        np.save('data_output/risk_ensemble_v', risk_var)
        np.save('data_output/resilience_field', resilience_field)
            
    def riskfield(self, filename, real_n, time_index):
        if not real_n == 'ensemble':
            kfield = self.kfields[real_n]
            alpha_v = 0.7
            field_c = np.load(f'data_output/cfields/cfield_{real_n}.npy')
            field_maxrisk = np.zeros(field_c.shape)
            c0 = field_c[0].max()
            field_maxrisk = np.where(field_c >= (self.mcl*c0), field_c/(self.mcl*c0), 0)

            for i in time_index:
                fig, ax = plt.subplots(figsize=(7,5))
                cmap = copy(plt.get_cmap('Reds'))
                def fmt(x, pos):
                    a, b = '{:.0e}'.format(x).split('e')
                    b = int(b)
                    return r'${} \times 10^{{{}}}$'.format(a, b)

                if not real_n == 'ensemble':
                    img = ax.imshow(kfield, cmap='jet', extent=[0,self.Lx/self.lambda_x,self.Ly/self.lambda_y,0],
                                    vmin=self.kmin, vmax=self.kmax)
                img = ax.imshow(field_maxrisk[i], cmap=cmap, extent=[0,self.Lx/self.lambda_x,self.Ly/self.lambda_y,0], 
                                vmin=1, vmax=np.ceil(field_maxrisk[i][:,:-8].max()), alpha=alpha_v)
                cbar = ax.figure.colorbar(img, ax=ax, fraction=0.041, pad=0.04, format=FuncFormatter(fmt))
                cbar.ax.tick_params(labelsize=14)
                cbar.ax.set_ylabel(r'$\rm{max}$ $c~/~\rm{mcl}$', fontsize=25, fontname='Arial', labelpad=10)
                cbar.solids.set_edgecolor("face")
                ax.scatter(self.observation_wells.T[0]/self.lambda_x, self.observation_wells.T[1]/self.lambda_y, 
                           color='k', marker='^' ,s=50)                  

                ax.xaxis.set_major_locator(LinearLocator(5))
                ax.xaxis.set_minor_locator(LinearLocator(21))
                ax.yaxis.set_major_locator(LinearLocator(5))
                ax.yaxis.set_minor_locator(LinearLocator(21))

                rectangle2 = plt.Rectangle((self.target_xl/self.lambda_x,self.target_yl/self.lambda_y), 
                                          (self.target_xu-self.target_xl)/self.lambda_x, 
                                          (self.target_yu-self.target_yl)/self.lambda_y, 
                                          fc='k', fill=None, linewidth=1.5)
                plt.gca().add_patch(rectangle2)            

                plt.xlim(0,self.Lx/self.lambda_x)
                plt.ylim(0,self.Ly/self.lambda_y)
                plt.xticks(fontsize=15, fontname='Arial')
                plt.yticks(fontsize=15, fontname='Arial')
                plt.xlabel(r'$x~/~\lambda_x$', fontsize=25, fontname='Arial', labelpad=5)
                plt.ylabel(r'$y~/~\lambda_y$', fontsize=25, fontname='Arial', labelpad=5)
                plt.text(5/self.lambda_x, 5/self.lambda_y, f'$t={i*self.dt}$', fontsize=15, color='k')
                plt.tight_layout()
                plt.savefig(f'figures/{filename}_{i}.png',dpi=200, bbox_inches='tight')            
        else:
            alpha_v = 1
            field_maxrisk = np.load(f'data_output/risk_ensemble.npy')
            risk_var = np.load('data_output/risk_ensemble_v.npy', )
            
            for i in time_index:
                fig, ax = plt.subplots(figsize=(7,5))
                cmap = copy(plt.get_cmap('Reds'))
                def fmt(x, pos):
                    a, b = '{:.0e}'.format(x).split('e')
                    b = int(b)
                    return r'${} \times 10^{{{}}}$'.format(a, b)

                img = ax.imshow(field_maxrisk[i], cmap=cmap, extent=[0,self.Lx/self.lambda_x,self.Ly/self.lambda_y,0], 
                                vmin=0, vmax=1, alpha=alpha_v)
                cbar = ax.figure.colorbar(img, ax=ax, fraction=0.041, pad=0.04, format=FuncFormatter(fmt))
                cbar.ax.tick_params(labelsize=14)
                cbar.ax.set_ylabel(r'$\left<\Psi\right>$', fontsize=25, fontname='Arial', labelpad=10)
                cbar.solids.set_edgecolor("face")
                ax.scatter(self.observation_wells.T[0]/self.lambda_x, self.observation_wells.T[1]/self.lambda_y, 
                           color='k', marker='^' ,s=50)            
                

                ax.xaxis.set_major_locator(LinearLocator(5))
                ax.xaxis.set_minor_locator(LinearLocator(21))
                ax.yaxis.set_major_locator(LinearLocator(5))
                ax.yaxis.set_minor_locator(LinearLocator(21))

                rectangle2 = plt.Rectangle((self.target_xl/self.lambda_x,self.target_yl/self.lambda_y), 
                                          (self.target_xu-self.target_xl)/self.lambda_x, 
                                          (self.target_yu-self.target_yl)/self.lambda_y, 
                                          fc='k', fill=None, linewidth=1.5)
                plt.gca().add_patch(rectangle2)            

                plt.xlim(0,self.Lx/self.lambda_x)
                plt.ylim(0,self.Ly/self.lambda_y)
                plt.xticks(fontsize=15, fontname='Arial')
                plt.yticks(fontsize=15, fontname='Arial')
                plt.xlabel(r'$x~/~\lambda_x$', fontsize=25, fontname='Arial', labelpad=5)
                plt.ylabel(r'$y~/~\lambda_y$', fontsize=25, fontname='Arial', labelpad=5)
                plt.text(5/self.lambda_x, 5/self.lambda_y, f'$t={i*self.dt}$', fontsize=15, color='k')
                plt.tight_layout()
                plt.savefig(f'figures/{filename}_{i}.png',dpi=200, bbox_inches='tight')
                
                fig, ax = plt.subplots(figsize=(7,5))
                cmap = copy(plt.get_cmap('Purples'))
                def fmt(x, pos):
                    a, b = '{:.0e}'.format(x).split('e')
                    b = int(b)
                    return r'${} \times 10^{{{}}}$'.format(a, b)

                img = ax.imshow(risk_var[i], cmap=cmap, extent=[0,self.Lx/self.lambda_x,self.Ly/self.lambda_y,0], 
                                vmin=0, vmax=np.max(risk_var[i][:,:-self.lambda_x]), alpha=alpha_v)
                   
                cbar = ax.figure.colorbar(img, ax=ax, fraction=0.041, pad=0.04, format=FuncFormatter(fmt))
                cbar.ax.tick_params(labelsize=14)
                cbar.ax.set_ylabel(r'$\sigma^2_{\Psi}$', fontsize=25, fontname='Arial', labelpad=10)
                cbar.solids.set_edgecolor("face")
                ax.scatter(self.observation_wells.T[0]/self.lambda_x, self.observation_wells.T[1]/self.lambda_y, 
                           color='k', marker='^' ,s=50)            

                ax.xaxis.set_major_locator(LinearLocator(5))
                ax.xaxis.set_minor_locator(LinearLocator(21))
                ax.yaxis.set_major_locator(LinearLocator(5))
                ax.yaxis.set_minor_locator(LinearLocator(21))

                rectangle2 = plt.Rectangle((self.target_xl/self.lambda_x,self.target_yl/self.lambda_y), 
                                          (self.target_xu-self.target_xl)/self.lambda_x, 
                                          (self.target_yu-self.target_yl)/self.lambda_y, 
                                          fc='k', fill=None, linewidth=1.5)
                plt.gca().add_patch(rectangle2)            

                plt.xlim(0,self.Lx/self.lambda_x)
                plt.ylim(0,self.Ly/self.lambda_y)
                plt.xticks(fontsize=15, fontname='Arial')
                plt.yticks(fontsize=15, fontname='Arial')
                plt.xlabel(r'$x~/~\lambda_x$', fontsize=25, fontname='Arial', labelpad=5)
                plt.ylabel(r'$y~/~\lambda_y$', fontsize=25, fontname='Arial', labelpad=5)
                plt.text(5/self.lambda_x, 5/self.lambda_y, f'$t={i*self.dt}$', fontsize=15, color='k')
                plt.tight_layout()
                plt.savefig(f'figures/{filename}_v_{i}.png',dpi=200, bbox_inches='tight')

        
    def resiliencefield(self, filename, real_n):        
        if not real_n == 'ensemble':
            kfield = self.kfields[real_n]
            field_resilience = np.load(f'data_output/resilience_field.npy')[real_n]
            alpha_v = 0.7
        else:
            field_resilience = np.load(f'data_output/resilience_field.npy').mean(axis=0)
            field_resilience_var = np.load(f'data_output/resilience_field.npy').var(axis=0)
            alpha_v = 1
        
        fig, ax = plt.subplots(figsize=(7,5))
        cmap = copy(plt.get_cmap('Blues'))
        def fmt(x, pos):
            a, b = '{:.0e}'.format(x).split('e')
            b = int(b)
            return r'${} \times 10^{{{}}}$'.format(a, b)

        if not real_n == 'ensemble':
            img = ax.imshow(kfield, cmap='jet', extent=[0,self.Lx/self.lambda_x,self.Ly/self.lambda_y,0], 
                            vmin=self.kmin, vmax=self.kmax)
        img = ax.imshow(field_resilience, cmap=cmap, extent=[0,self.Lx/self.lambda_x,self.Ly/self.lambda_y,0], 
                        vmin=0, vmax=np.ceil(field_resilience[:,:-8].max()), alpha=0.7)
        cbar = ax.figure.colorbar(img, ax=ax, fraction=0.041, pad=0.04, format=FuncFormatter(fmt))
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.set_ylabel(r'$\left<R_L\right>$', fontsize=25, fontname='Arial', labelpad=10)
        cbar.solids.set_edgecolor("face")
        ax.scatter(self.observation_wells.T[0]/self.lambda_x, self.observation_wells.T[1]/self.lambda_y, 
                   color='k', marker='^' ,s=50)          
        
        ax.xaxis.set_major_locator(LinearLocator(5))
        ax.xaxis.set_minor_locator(LinearLocator(21))
        ax.yaxis.set_major_locator(LinearLocator(5))
        ax.yaxis.set_minor_locator(LinearLocator(21))
        
        rectangle2 = plt.Rectangle((self.target_xl/self.lambda_x,self.target_yl/self.lambda_y), 
                                  (self.target_xu-self.target_xl)/self.lambda_x, 
                                  (self.target_yu-self.target_yl)/self.lambda_y, 
                                  fc='k', fill=None, linewidth=1.5)
        plt.gca().add_patch(rectangle2)        
        
        plt.xlim(0,self.Lx/self.lambda_x)
        plt.ylim(0,self.Ly/self.lambda_y)
        plt.xticks(fontsize=15, fontname='Arial')
        plt.yticks(fontsize=15, fontname='Arial')
        plt.xlabel(r'$x~/~\lambda_x$', fontsize=25, fontname='Arial', labelpad=5)
        plt.ylabel(r'$y~/~\lambda_y$', fontsize=25, fontname='Arial', labelpad=5)

        plt.tight_layout()
        plt.savefig(f'figures/{filename}.png',dpi=200, bbox_inches='tight')
        
        if real_n == 'ensemble':
            fig, ax = plt.subplots(figsize=(7,5))
            cmap = copy(plt.get_cmap('Purples'))
            def fmt(x, pos):
                a, b = '{:.0e}'.format(x).split('e')
                b = int(b)
                return r'${} \times 10^{{{}}}$'.format(a, b)

            img = ax.imshow(field_resilience_var, cmap=cmap, extent=[0,self.Lx/self.lambda_x,self.Ly/self.lambda_y,0], 
                            vmin=0, vmax=np.ceil(field_resilience_var[:,:-8].max()), alpha=0.7)
            cbar = ax.figure.colorbar(img, ax=ax, fraction=0.041, pad=0.04, format=FuncFormatter(fmt))
            cbar.ax.tick_params(labelsize=14)
            cbar.ax.set_ylabel(r'$\sigma^2_{R_L}$', fontsize=25, fontname='Arial', labelpad=10)
            cbar.solids.set_edgecolor("face")
            ax.scatter(self.observation_wells.T[0]/self.lambda_x, self.observation_wells.T[1]/self.lambda_y, 
                       color='k', marker='^' ,s=50)          

            ax.xaxis.set_major_locator(LinearLocator(5))
            ax.xaxis.set_minor_locator(LinearLocator(21))
            ax.yaxis.set_major_locator(LinearLocator(5))
            ax.yaxis.set_minor_locator(LinearLocator(21))

            rectangle2 = plt.Rectangle((self.target_xl/self.lambda_x,self.target_yl/self.lambda_y), 
                                      (self.target_xu-self.target_xl)/self.lambda_x, 
                                      (self.target_yu-self.target_yl)/self.lambda_y, 
                                      fc='k', fill=None, linewidth=1.5)
            plt.gca().add_patch(rectangle2)        

            plt.xlim(0,self.Lx/self.lambda_x)
            plt.ylim(0,self.Ly/self.lambda_y)
            plt.xticks(fontsize=15, fontname='Arial')
            plt.yticks(fontsize=15, fontname='Arial')
            plt.xlabel(r'$x~/~\lambda_x$', fontsize=25, fontname='Arial', labelpad=5)
            plt.ylabel(r'$y~/~\lambda_y$', fontsize=25, fontname='Arial', labelpad=5)

            plt.tight_layout()
            plt.savefig(f'figures/{filename}.png',dpi=200, bbox_inches='tight')
            

    def eta_postprocessing(self):
        cell_number = self.Lx*self.Ly
        sflow = np.zeros((self.n_realization,self.Ly,self.Lx))

        for i in range(self.n_realization):
            velocities = []
            f = open(f'tmp/model-{i}.ftl', 'r')
            for line in f:
                if line[3] == 'X':
                    for j in range(0, cell_number, 3):
                        line = f.readline()
                        e1, e2, e3 = np.asarray(line.split(), dtype=float)
                        velocities.append(e1)
                        velocities.append(e2)
                        velocities.append(e3)
            f.close()
            sflow[i] = np.asarray(velocities).reshape((self.Ly,self.Lx))
        
        np.save('data_output/sflow.npy',sflow)
        
        eta = sflow[:,self.source_yl:self.source_yu,self.source_xu].mean(axis=(1))/(1/self.Lx*np.exp(np.log(self.Kg)))
        
        np.save('data_output/eta', eta)
        
    def maxriskresilience_postprocessing(self):
        cfield_all = np.load('data_output/cfields/cfield_all.npy')
        resilience_field = np.load('data_output/resilience_field.npy')
        maxrisk = np.zeros(self.n_realization)
        maxresilience = np.zeros(self.n_realization)
        
        for real_n in range(self.n_realization):
            field_c = cfield_all[real_n]
            field_maxrisk = np.zeros(field_c.shape)    
            field_reliability = np.zeros(field_c.shape)
            nt = len(field_c)
            c0 = field_c[0].max()
            
            field_maxrisk = np.where(field_c >= (self.mcl*c0), field_c/(self.mcl*c0), 0)
            field_resilience = resilience_field[real_n]

            maxrisk[real_n] = field_maxrisk[:,self.target_yl:self.target_yu,self.target_xl:self.target_xu].max()
            maxresilience[real_n] = field_resilience[self.target_yl:self.target_yu,self.target_xl:self.target_xu].max()
        
        np.save('data_output/maxrisk', maxrisk)
        np.save('data_output/maxresilience', maxresilience)
        
    def eta_rr(self, filename, real_n):
        eta = np.load('data_output/eta.npy')
        maxrisk = np.load('data_output/maxrisk.npy')
        maxresilience = np.load('data_output/maxresilience.npy')

        fig, ax = plt.subplots(figsize=(6,5))

        interp = np.linspace(0,np.ceil(eta.max()),100)
        plt.plot(interp, 293.798518*scipy.special.erf(2.375823*interp) + 449.030143, color='b', linewidth=2, label='Trend line')
        plt.scatter(eta, maxrisk, color='gray', s=25, alpha=0.8)

        if not real_n == 'ensemble':
            plt.scatter(eta[real_n], maxrisk[real_n], color='r', s=100, alpha=1, label=f'Realization {real_n}')

        ax.xaxis.set_major_locator(LinearLocator(5))
        ax.xaxis.set_minor_locator(LinearLocator(21))
        ax.yaxis.set_major_locator(LinearLocator(5))
        ax.yaxis.set_minor_locator(LinearLocator(21))

        plt.xlim(0,np.ceil(eta.max()))
        plt.ylim(0,np.ceil(maxrisk.max()*1.05))
        plt.xticks(fontsize=15, fontname='Arial')
        plt.yticks(fontsize=15, fontname='Arial')
        plt.xlabel(r'$\eta$', fontsize=25, fontname='Arial', labelpad=5)
        plt.ylabel(r'$c_{max}~/~\rm{mcl}$', fontsize=25, fontname='Arial', labelpad=5)

        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'figures/{filename}.png',dpi=200, bbox_inches='tight')
        plt.show()
        
        fig, ax = plt.subplots(figsize=(6,5))
        interp = np.linspace(0,np.ceil(eta.max()),100)
        plt.plot(interp, 292.155804 - 125.685574*np.log(interp), color='b', linewidth=2, label='Trend line')
        plt.scatter(eta, maxresilience, color='gray', s=25, alpha=0.8)

        if not real_n == 'ensemble':
            plt.scatter(eta[real_n], maxresilience[real_n], color='r', s=100, alpha=1, label=f'Realization {real_n}')

        ax.xaxis.set_major_locator(LinearLocator(5))
        ax.xaxis.set_minor_locator(LinearLocator(21))
        ax.yaxis.set_major_locator(LinearLocator(5))
        ax.yaxis.set_minor_locator(LinearLocator(21))

        plt.xlim(0,np.ceil(eta.max()))
        plt.ylim(0,np.ceil(maxresilience.max()*1.05))
        plt.xticks(fontsize=15, fontname='Arial')
        plt.yticks(fontsize=15, fontname='Arial')
        plt.xlabel(r'$\eta$', fontsize=25, fontname='Arial', labelpad=5)
        plt.ylabel(r'$R_L$', fontsize=25, fontname='Arial', labelpad=5)

        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'figures/{filename}.png',dpi=200, bbox_inches='tight')
        plt.show()
    
    def well_postprocessing(self):
        cfield_all = np.load('data_output/cfields/cfield_all.npy')
        obwells_conc = []
        for i in range(len(self.observation_wells)):
            obwells_conc.append(cfield_all[:,:,self.observation_wells.T[1][i],self.observation_wells.T[0][i]])
        obwells_conc = np.asarray(obwells_conc)
        obwells_maxconc = obwells_conc.max(axis=2)
        np.save('data_output/obwells_maxconc', obwells_maxconc)
    
    def cdf_maxconc(self, filename):
        obwells_maxconc = np.load('data_output/obwells_maxconc.npy' )
        
        for i in range(len(obwells_maxconc)):
            fig, ax = plt.subplots(figsize=(6,5))
            obdata = obwells_maxconc[i][obwells_maxconc[i]!=0]

            numerator = len(obwells_maxconc[i][obwells_maxconc[i]==0])
            denominator = len(obwells_maxconc[i])
            bottom_p = numerator/denominator

            hist_result, interval = np.histogram(obdata, density=True, bins=100)
            dx = interval[1] - interval[0]
            cdf_result = np.cumsum(hist_result)*dx
            cdf = (cdf_result*(1-bottom_p)) + bottom_p
            survival_p = 1 - cdf
            midpoint = (interval[:-1]+interval[1:])/2
            plt.plot(midpoint,survival_p, label=f'well {i+1}', linewidth=2,)
        
            plt.xscale('log')
            mcl_index = np.argmin(np.abs(midpoint-self.mcl))

            print(f'the probability of the maximum concentration at the well {i+1} over {round(midpoint[mcl_index],3)} is {round(survival_p[mcl_index],3)}')

            plt.ylim(0, np.max([survival_p]))

            plt.xticks(fontsize=15, fontname='Arial')
            plt.yticks(fontsize=15, fontname='Arial')
            plt.xlabel(r'$c_{max}$', fontsize=25, fontname='Arial', labelpad=5)
            plt.ylabel(r'$S$', fontsize=25, fontname='Arial', labelpad=5)
            plt.tick_params(which="major", direction="in", right=True, top=True, length=5, pad=7)
            plt.tick_params(which="minor", direction="in", right=True, top=True, length=3)
            plt.legend(fontsize=12, loc=3)
            plt.savefig(f'figures/{filename}_{i}.png',dpi=200, bbox_inches='tight')
            plt.show()
