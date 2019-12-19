import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy as sp
import tkinter as tk
import PIL

class click_stack(object):
    def __init__(self,plateifu):
        #####
        self.dir1='/home/ashley/Desktop/GW_张彬彬/manga_pipeline/result/'
        ####
        self.plateifu=plateifu
        
        self.access_data()
        self.tk_part()

        
    def access_data(self):
        with fits.open('/media/ashley/zsash/MPL-7-LOGCUBE/10001/12701/manga-10001-12701-LOGCUBE-HYB10-GAU-MILESHC.fits.gz') as hdulist:    
            hdulist.info()
            self.wave = hdulist['WAVE'].data
            self.flux_t = np.transpose(hdulist['FLUX'].data, axes=(1,2,0))
            self.ivar_t = np.transpose(hdulist['IVAR'].data, axes=(1,2,0))
            self.flux_header=hdulist['FLUX'].header
            self.filename=self.dir1+'test.png'
            plt.imsave(self.filename, self.flux_t[0], cmap=plt.cm.gray, vmin=np.min(self.flux_t[0]), vmax=np.max(self.flux_t[0]), origin="lower")
        
        with fits.open('/media/ashley/zsash/MPL-7/10001/12701/manga-10001-12701-MAPS-HYB10-GAU-MILESHC.fits.gz') as mapf:
            self.flux_map = mapf['EMLINE_GFLUX'].data
            self.ivar_map = mapf['EMLINE_GFLUX_IVAR'].data
            self.mask_map = mapf['EMLINE_GFLUX_MASK'].data
            self.ellcoo = mapf['SPX_ELLCOO'].data[1]
            self.stellar_vel = mapf['STELLAR_VEL'].data
            self.spx_coo=mapf['SPX_SKYCOO'].data
            self.file_map=self.dir1+'Ha_map.png'
            plt.imsave(self.file_map,self.flux_map[18],cmap=plt.cm.gray, vmin=np.min(self.flux_map[18]), vmax=np.max(self.flux_map[18]), origin="lower",dpi=300)
            """
            ??? to do:
            stop mouse click
            ???? future select region
            
            """
    def stack(self):
        a=np.zeros(self.flux_map[18].shape)
        for i in range(0,len(self.location_x)):
            a[self.location_x[i]][self.location_y[i]]=1
        mask=(a>0)
        self.flux=self.flux_t[mask]
        self.vel=self.stellar_vel[mask]
        self.z=(self.vel/3e5)[mask]
        for  j in range(0,len(self.flux)):
            wave_rest=self.wave/(1+self.z[j])
            flux=np.interp(self.wave,wave_rest,self.flux[j])
            if flux.shape>0:
                self.flux_stack+=flux
        plt.plot(self.wave,self.flux_stack,dpi=300)
        plt.savefig(self.dir1+'stack.png',format='png',dpi=300)
            
    def tk_part(self):
        #Try tk stuff
        root = tk.Tk()
        
        #setting up a tkinter canvas
        frame = tk.Frame(root, bd=2, relief=tk.SUNKEN)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        canvas = tk.Canvas(frame, bd=0)
        canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        frame.pack(fill=tk.BOTH,expand=1)
        
       
        #Add image
        hold1=tk.PhotoImage(file=self.file_map)
        canvas.create_image(0,0,image=hold1,anchor='nw')
        
        
        #function to be called when mouse is clicked
        self.location_x=[]
        self.location_y=[]
        def printcoords(event):
            #outputting x and y coords to console
            print (event.x,event.y)
            self.location_x.append(event.x)
            self.location_y.append(event.y)
        #mouseclick event
        canvas.bind("<Button 1>",printcoords)
        
        self.stack()
        hold2=tk.PhotoImage(file=self.dir1+'stack.png')
        canvas.create_image(1,0,image=hold2,anchor='nw')
        root.mainloop()

if __name__=='__main__':
    click_stack('10001-12701')