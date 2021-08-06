#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 10:20:31 2019

@author: ashley
"""

#    
#if __name__=='__main__':
#    plateifu=sys.argv[1]
#    access_data(plateifu)

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pdfkit
import ppxf as ppxf_package
import ppxf.miles_util as lib
import ppxf.ppxf_util as util
import ppxf.ppxfgas as gas
import ppxf.ppxfstellar as stellar
import scipy.stats as stats
import sys
import wget

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from matplotlib import cm
from ppxf.ppxf import ppxf
from PIL import Image
from PyPDF2 import PdfFileMerger




np.seterr(divide='ignore', invalid='ignore')

def ppxf_example_kinematics_sdss(dirfile, galaxy, lam_gal, plateifu, mask,
                                 noise, redshift):

    ppxf_dir = os.path.dirname(os.path.realpath(ppxf_package.__file__))

    z = redshift
    lam_gal = lam_gal
    mask = mask

    c = 299792.458
    frac = lam_gal[1] / lam_gal[0]
    # dlamgal = (frac - 1) * lam_gal
    a = np.full((1, 4563), 2.76)
    fwhm_gal = a[0][mask]

    velscale = np.log(frac) * c

    vazdekis = glob.glob(ppxf_dir + '/miles_models/Mun1.30Z*.fits')
    fwhm_tem = 2.51
    hdu = fits.open(vazdekis[0])
    ssp = hdu[0].data
    h2 = hdu[0].header
    lam_temp = h2['CRVAL1'] + h2['CDELT1'] * np.arange(h2['NAXIS1'])
    lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]
    sspNew = util.log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
    templates = np.empty((sspNew.size, len(vazdekis)))
    fwhm_gal = np.interp(lam_temp, lam_gal, fwhm_gal)
    hdu.close()

    fwhm_dif = np.sqrt((fwhm_gal**2 - fwhm_tem**2).clip(0))
    sigma = fwhm_dif / 2.355 / h2['CDELT1']  # Sigma difference in pixels

    for j, fname in enumerate(vazdekis):
        hdu = fits.open(fname)
        ssp = hdu[0].data
        ssp = util.gaussian_filter1d(
            ssp, sigma)  # perform convolution with variable sigma
        sspNew = util.log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
        templates[:, j] = sspNew / np.median(sspNew)  # Normalizes templates
        hdu.close()

    c = 299792.458
    dv = np.log(lam_temp[0] / lam_gal[0]) * c  # km/s
    goodpixels = util.determine_goodpixels(np.log(lam_gal), lamRange_temp, z)

    vel = c * np.log(1 + z)  # eq.(8) of Cappellari (2017)
    start = [vel, 200.]  # (km/s), starting guess for [V, sigma]


    pp = stellar.ppxf(dirfile,
                      templates,
                      galaxy,
                      noise,
                      velscale,
                      start,
                      z,
                      goodpixels=goodpixels,
                      plot= True,
                      moments=4,
                      degree=12,
                      vsyst=dv,
                      clean=False,
                      lam=lam_gal)

    return pp.bestfit, pp.lam


def emission(dirfile, w1, f1, redshift, plateifu, tie_balmer, limit_doublets):
    
    ppxf_dir = os.path.dirname(os.path.realpath(ppxf_package.__file__))
    
    z = redshift
    flux = f1
    galaxy = flux
    wave = w1

    wave *= np.median(util.vac_to_air(wave) / wave)

    noise = np.full_like(galaxy,
                         0.01635)  # Assume constant noise per pixel here

    c = 299792.458  # speed of light in km/s
    velscale = c * np.log(wave[1] / wave[0])  # eq.(8) of Cappellari (2017)
    # SDSS has an approximate instrumental resolution FWHM of 2.76A.
    FWHM_gal = 2.76

    # ------------------- Setup templates -----------------------

    pathname = ppxf_dir + '/miles_models/Mun1.30*.fits'

    # The templates are normalized to mean=1 within the FWHM of the V-band.
    # In this way the weights and mean values are light-weighted quantities
    miles = lib.miles(pathname, velscale, FWHM_gal)

    # reg_dim = miles.templates.shape[1:]

    # regul_err = 0.013  # Desired regularization error

    lam_range_gal = np.array([np.min(wave), np.max(wave)]) / (1 + z)
    gas_templates, gas_names, line_wave = util.emission_lines(
        miles.log_lam_temp,
        lam_range_gal,
        FWHM_gal,
        tie_balmer=tie_balmer,
        limit_doublets=limit_doublets)

    templates = gas_templates

    c = 299792.458
    dv = c * (miles.log_lam_temp[0] - np.log(wave[0])
              )  # eq.(8) of Cappellari (2017)
    vel = c * np.log(1 + z)  # eq.(8) of Cappellari (2017)
    start = [vel, 180.]  # (km/s), starting guess for [V, sigma]

    #     n_temps = stars_templates.shape[1]
    n_forbidden = np.sum(["[" in a
                          for a in gas_names])  # forbidden lines contain "[*]"
    n_balmer = len(gas_names) - n_forbidden

    component = [0] * n_balmer + [1] * n_forbidden
    gas_component = np.array(
        component) >= 0  # gas_component=True for gas templates

    moments = [2, 2]

    start = [start, start]

    gas_reddening = 0 if tie_balmer else None


    pp = gas.ppxf(dirfile,
                  templates,
                  galaxy,
                  noise,
                  velscale,
                  start,
                  plot=False,
                  moments=moments,
                  degree=-1,
                  mdegree=10,
                  vsyst=dv,
                  lam=wave,
                  clean=False,
                  component=component,
                  gas_component=gas_component,
                  gas_names=gas_names,
                  gas_reddening=gas_reddening)

    plt.figure()
    plt.clf()
    pp.plot()
    return pp.bestfit, pp.lam


def move_continuum(wave, z, width=800):
    """
    Generates a list of goodpixels to mask a given set of gas emission
    lines. This is meant to be used as input for PPXF.

    :param logLam: Natural logarithm np.log(wave) of the wavelength in
        Angstrom of each pixel of the log rebinned *galaxy* spectrum.
    :param lamRangeTemp: Two elements vectors [lamMin2, lamMax2] with the minimum
        and maximum wavelength in Angstrom in the stellar *template* used in PPXF.
    :param z: Estimate of the galaxy redshift.
    :return: vector of goodPixels to be used as input for pPXF

    """
    #                     -----[OII]-----   Hbeta   -----[OIII]-----   [OI]    -----[NII]-----   Halpha
    lines = np.array([
        3726.03, 3728.82, 4861.33, 4958.92, 5006.84, 6548.03, 6583.41, 6562.80
    ])
    dv = np.full_like(lines, width)
    c = 299792.458

    flag = False
    for line, dvj in zip(lines, dv):
        flag |= (wave > line*(1 + z)*(1 - dvj/(2*c))) \
              & (wave < line*(1 + z)*(1 + dvj/(2*c)))
    return flag







class manga(object):
    """
    """
    def __init__(self,plateifu):
        
        self.dir1='/media/nju/project/mangawork/manga/spectro/redux/v2_5_3/'
        self.dir2='/media/nju/project/mangawork/manga/spectro/analysis/v2_5_3/2.3.0/HYB10-MILESHC-MILESHC/'
        self.dir3 = '/home/nju/Desktop/GW_张彬彬/manga_pipeline/result/'
        
        self.plateifu=plateifu
        self.access_data()
        self.group()
        self.rest_frame()
        self.bpt()
        self.pipe3d()        
        self.ppxf_fitting()
        self.image()
        self.m_z_relation()
        self.print_pdf()

        
    def access_data(self):

        self.plate=self.plateifu.split('-')[0]
        self.ifu=self.plateifu.split('-')[1]
        self.f=self.dir1+str(self.plate)+'/stack/'+'manga-'+self.plateifu+'-LOGCUBE.fits.gz'
        self.f1=self.dir2+str(self.plate)+'/'+str(self.ifu)+'/manga-'+self.plateifu+'-MAPS-HYB10-MILESHC-MILESHC.fits.gz'

        if os.path.exists(self.f) and not os.path.exists(self.f1):
            print('file not existed: {0}'.format(self.f1))
            url_f1='https://data.sdss.org/sas/dr16/manga/spectro/analysis/v2_4_3/2.2.1/HYB10-GAU-MILESHC/'+str(self.plate)+'/'+str(self.ifu)+'/manga-'+self.plateifu+'-MAPS-HYB10-GAU-MILESHC.fits.gz'
            self.f1=wget.download(url_f1,out=self.dir3)
        elif os.path.exists(self.f1) and not os.path.exists(self.f):
            print('file not existed: {0}'.format(self.f))
            url_f='https://data.sdss.org/sas/dr16/manga/spectro/redux/v2_4_3/'+self.plate+'/stack/manga-'+self.plateifu+'-LOGCUBE.fits.gz'
            self.f=wget.download(url_f,out=self.dir3)
        elif not os.path.exists(self.f1) and not os.path.exists(self.f):
            print('file not existed: {0},{1}'.format(self.f,self.f1))
            
            url_f1='https://data.sdss.org/sas/dr16/manga/spectro/analysis/v2_4_3/2.2.1/HYB10-GAU-MILESHC/'+str(self.plate)+'/'+str(self.ifu)+'/manga-'+self.plateifu+'-MAPS-HYB10-GAU-MILESHC.fits.gz'
            print(url_f1)
            self.f1=wget.download(url_f1,out=self.dir3)
            url_f='https://data.sdss.org/sas/dr16/manga/spectro/redux/v2_4_3/'+self.plate+'/stack/manga-'+self.plateifu+'-LOGCUBE.fits.gz'
            print(url_f)
            self.f=wget.download(url_f,out=self.dir3)
        with fits.open(self.f) as cube, fits.open(self.f1) as mapf:
            self.wave = cube['WAVE'].data
            self.flux_header = cube['FLUX'].header
            """
            ??? axes sequence
            """
            self.flux_t = np.transpose(cube['FLUX'].data, axes=(2,1,0))
            self.ivar_t = np.transpose(cube['IVAR'].data, axes=(2,1,0))
            self.mask = np.transpose(cube['MASK'].data, axes=(2,1,0))

            self.flux_map = mapf['EMLINE_GFLUX'].data
            self.ivar_map = mapf['EMLINE_GFLUX_IVAR'].data
            self.mask_map = mapf['EMLINE_GFLUX_MASK'].data
            self.ellcoo = mapf['SPX_ELLCOO'].data[1]
            self.stellar_vel = mapf['STELLAR_VEL'].data
    def image(self):
        f_image=self.dir1+self.plate+'/images/'+self.ifu+'.png'
        img=Image.open(f_image)
        img1=Image.new('RGB', img.size, (255, 255, 255)) 
        img1.paste(img, mask=img.split()[3])               # paste using alpha channel as mask
        img1.save(self.dir3+'image.pdf', 'PDF',resolution=300)
        
        
    def bpt(self):
        """
        remove bad mask
        """
        mask_bad=(self.mask_map!=0) # !=0 remove
        self.flux_map=np.ma.array(self.flux_map,mask=mask_bad)
        self.ivar_map=np.ma.array(self.ivar_map,mask=mask_bad)
        
        """
        BPT diagram remove AGN
        Kauffmann 2003: log(OIII/hb)>0.61/(log(NII/Ha)-0.05)+1.3
        [oiii]5008,4960:13,12(5007)
        Hb4862:11
        NII6549,6585:17,19(6584)
        Ha6564:18
        """        
        self.x=np.log10(self.flux_map[19]/self.flux_map[18])
        self.y=np.log10(self.flux_map[13]/self.flux_map[11])
        self.mask_sf=(self.y<0.61/(self.x-0.05)+1.3)&(self.x<0.05)&(self.y<(0.61/(self.x-0.47)+1.19))
        self.mask_agn=(1-self.mask_sf).astype(np.bool)

        plt.figure(figsize=(12,12))
        ax1=plt.subplot(211)
        ax2=plt.subplot(223)
        ax3=plt.subplot(224)
        
        x_bpt = np.arange(np.min(self.x)-0.01, 0.2, 0.02)
        y_bpt = 0.61/(x_bpt-0.47)+1.19
        x_bpt1 = np.arange(np.min(self.x)-0.01, -0.2, 0.02)
        y_bpt1 = 0.61/(x_bpt1-0.05)+1.3
        ax1.plot(x_bpt, y_bpt, 'r')
        ax1.plot(x_bpt1, y_bpt1, 'k:')
        ax1.scatter(self.x[self.mask_sf],self.y[self.mask_sf],s=2,color='orange',label='star forming')
        ax1.scatter(self.x[self.mask_agn],self.y[self.mask_agn],s=2,color='dodgerblue',label='composite region or AGN')
        ax1.legend()
        ax1.set_xlabel('log(NII/Ha)')
        ax1.set_ylabel('log([OIII]/Hb)')
        ax1.set_ylim(-1,1.5)
        ax1.set_title(self.plateifu)
        
        dx = self.flux_header['CD1_1'] * 3600.  # deg to arcsec
        dy = self.flux_header['CD2_2'] * 3600.  # deg to arcsec
        x_extent = (np.array([0., self.mask_sf.shape[0]]) - (self.mask_sf.shape[0] - self.x_center)) * dx * (-1)
        y_extent = (np.array([0., self.mask_sf.shape[1]]) - (self.mask_sf.shape[1] - self.y_center)) * dy
        self.extent = [x_extent[0], x_extent[1], y_extent[0], y_extent[1]]
        print(self.extent)
        ax2.imshow(self.mask_agn,extent=self.extent, cmap=cm.YlGnBu_r, vmin=0.1, vmax=1, origin='lower', interpolation='none')
#        ax2.set_colorbar(label=self.flux_header['BUNIT'])
        ax2.set_xlabel('arcsec')
        ax2.set_ylabel('arcsec')
        ax2.set_title('composite region or AGN')
        
        ax3.imshow(self.mask_sf,extent=self.extent, cmap=cm.YlGnBu_r, vmin=0.1, vmax=1, origin='lower', interpolation='none')
        ax3.set_xlabel('arcsec')
        ax3.set_ylabel('arcsec')
        ax3.set_title('star forming region')
        plt.savefig(self.dir3+'bpt.pdf')

        
            
        
    def group(self):
        self.drp='/media/nju/project/mangawork/manga/spectro/redux/v2_5_3/drpall-v2_5_3.fits'
        with fits.open(self.drp) as f_drp:
            data_drp=f_drp[1].data
            ifu_drp=data_drp['plateifu']
            ra_drp=data_drp['objra']
            dec_drp=data_drp['objdec']
            z_drp=data_drp['z']
            
            index=np.where(ifu_drp==self.plateifu)[0][0]
            print(ifu_drp[index],index)
            ra_drp1=ra_drp[index]
            dec_drp1=dec_drp[index]
            z_drp1=z_drp[index]
            self.z=z_drp1
            print(self.z)
            
            manga=SkyCoord(ra=ra_drp1*u.deg,dec=dec_drp1*u.deg)
            catalog=SkyCoord(ra=ra_drp*u.deg,dec=dec_drp*u.deg)
            dis=manga.separation(catalog).arcsec
            
            
            index1=np.where(np.logical_and(dis>=3, dis<=50))[0]
            if len(index1)>0:
                print('{0} have {1} galaxies nearby : '.format(self.plateifu,len(index1)))
                for i in range(0,len(index1)):
                    print('{0} : distance(arcsec) : {1}'.format(ifu_drp[index1[i]],dis[index1[i]]))
            else:
                print('{0} have no galaxies nearby with manga observation.'.format(self.plateifu))
                
    def rest_frame(self):
        self.x_center = np.int(self.flux_header['CRPIX1']) - 1
        self.y_center = np.int(self.flux_header['CRPIX2']) - 1
        
        plt.figure(figsize=(12,12))
        plt.plot(self.wave/(1+self.z), self.flux_t[self.x_center, self.y_center])
        plt.xlabel('$rest-frame \ \lambda \, [\AA]  $ ')
        plt.ylabel(self.flux_header['BUNIT'])
        plt.title('center')
        plt.savefig(self.dir3+self.plateifu+'_rest_frame.pdf')

        
    def pipe3d(self):
#        url='https://data.sdss.org/sas/dr16/manga/spectro/pipe3d/v2_4_3/2.4.3/'+self.plate+'/manga-'+self.plateifu+'.Pipe3D.cube.fits.gz'
#        file1=wget.download(url,out=self.dir3)
        file1='/home/nju/Desktop/GW_张彬彬/manga_pipeline/result/manga-10001-12701.Pipe3D.cube.fits.gz'
#        print(file1)
        with fits.open(file1) as f_pipe3d:
            self.data=f_pipe3d[1].data
            
        
        
    def m_z_relation(self):
        '''
        D02
        '''
        # self.metal=self.x*0.73+9.12
        plt.figure(figsize=(12,12))
        ax1=plt.subplot(221)
        ax2=plt.subplot(222)
        ax3=plt.subplot(212)
        
        ax1.imshow(self.data[9],extent=self.extent,cmap=cm.YlGnBu_r, vmin=np.min(self.data[9]), vmax=np.max(self.data[9]), origin='lower', interpolation='none')
        ax1.set_title('metallicity')
        
        print(self.data[9].shape,self.data[19].shape)
        ax2.imshow(self.data[19],extent=self.extent,cmap=cm.YlGnBu_r, vmin=np.min(self.data[19]), vmax=np.max(self.data[19]), origin='lower', interpolation='none')
        ax2.set_title('stellar mass density M_{\dot}/spaxel^{2}$')
        mask=(self.data[19]>0)
        metal=self.data[9][mask]
        mass_density=self.data[19][mask]
        print(mass_density,metal)
        
        bin_means, bin_edges, binnumber=stats.binned_statistic(mass_density, metal,statistic='median',bins=10)
        bin_x=[]
        for n in range(0,len(bin_means)):
            bin_x.append(bin_edges[n] + bin_edges[n+1])
        
        ax3.scatter(bin_x,bin_means,s=8)
        ax3.set_xlabel('$stellar mass density M_{\odot}/spaxel^{2}$')
        ax3.set_ylabel('mass weighted metallicity')
        
        plt.savefig(self.dir3+'m-z.pdf',dpi=300)
        """ 
        ???why metallicity and y distribution is different
        """

    def ppxf_fitting(self):
        self.dirfile='/home/nju/Desktop/GW_张彬彬/manga_pipeline/result/'+self.plateifu+'.txt'
        self.dir3 = '/home/nju/Desktop/GW_张彬彬/manga_pipeline/result/'
    
        with open(self.dir3 + 'Assertion_error.txt', 'a+') as f_Assertionerror:
                
            if os.path.exists(self.dir3+self.plateifu+'_con_v2.fits'):
                os.remove(self.dir3+self.plateifu+'_con_v2.fits')
            if os.path.exists(self.dir3+self.plateifu+'_con_goodpixel.fits'):
                os.remove(self.dir3+self.plateifu+'_con_goodpixel.fits')

            mask1 = (self.wave / (1 + self.z) > 3540) & (self.wave /
                                                     (1 + self.z) < 7409)

            flux = self.flux_t[self.x_center, self.y_center][mask1]
            galaxy = flux / np.median(flux)
            wave = self.wave[mask1]
            error= 1/np.sqrt(self.ivar_t)
            noise = error[self.x_center, self.y_center][mask1]
            f1 = galaxy
            w1 = wave
            try:
                print(self.plateifu)
                f2, w2 = ppxf_example_kinematics_sdss( self.dirfile, galaxy, wave,self.plateifu, mask1, noise,self.z)
                """
                先看看信噪比》5：
                """
                mask_em = move_continuum(w1, self.z)
                flux_mask = (f1 - f2)[mask_em]
                w1_mask = w1[mask_em]
                noise_mask = noise[mask_em]
                snr_em = np.sum(flux_mask) / np.sqrt(np.sum(noise_mask**2))

                # plt.figure()
                # plt.plot(w1_mask,
                #          flux_mask,
                #          label='emission line mask %s' % snr_em)
                # plt.legend()
                # plt.savefig( self.dirfile[:-4] + '_snr_em.pdf', dpi=300)
                # plt.axis('off')
                # plt.close()
                ff, ww = emission( self.dirfile,
                                  w1,
                                  f1 - f2,
                                  self.z,
                                  self.plateifu,
                                  tie_balmer=True,
                                  limit_doublets=False)

                plt.figure(figsize=(12,12))
                plt.plot(ww, ff, label='fitting')
                plt.plot(ww, galaxy - f2, ":", label='stacking')
                plt.title(self.plateifu)
                plt.legend()
                plt.savefig(self.dirfile[:-4] + '_show.pdf', bbox_inches='tight')
                plt.axis('off')
                plt.close()
                
            except AssertionError:
                print(self.plateifu, file=f_Assertionerror)
                pass
   
    def print_pdf(self):
        pdfkit.from_file(self.dirfile,self.dir3+'1.pdf')
        pdfs=glob.glob(self.dir3+'*pdf')
                
        merger = PdfFileMerger()
        
        for pdf in pdfs:
            merger.append(pdf)
        
        merger.write(self.dir3+self.plateifu+".pdf")
        merger.close()
                
if __name__=='__main__':      
    plateifu=sys.argv[1]
    manga(plateifu)
    
    """
    ??? docker
    """
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
