# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:59:47 2024

@author: Chunlin Xu
"""

import numpy as np
import scipy.constants as C
import copy
import logging

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('kappa.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


hbar = C.Planck/(2*np.pi)
kB = C.Boltzmann
NA = C.Avogadro

class Material():
    def __init__(self, **kwgs):
        for key in kwgs.keys():
            self.__setattr__(key, kwgs[key])
        if 'Twfile' in self.__dict__.keys():
            self._load_Tw(self.Twfile)
            
        if 'pdosfile' in self.__dict__.keys():
            self._load_pdos(self.pdosfile)
        
        if 'mfp' in self.__dict__.keys():
            self.mfp = self.mfp/1.0e9
    
    def _load_pdos(self, pdosfile):
        data = np.loadtxt(pdosfile).T
        nu = data[0]
        pdos = data[1]
        self.nu_THz = nu
        self.w_THz = nu*2*np.pi
        self.nu = nu*1.0e12
        self.w = self.nu*2*np.pi
        self.pdos = pdos

    def _load_Tw(self, Twfile):
        data = np.loadtxt(Twfile).T
        nu = data[0]
        Tw = data[1]
        self.nu_THz = nu
        self.w_THz = nu*2*np.pi
        self.nu = nu*1.0e12
        self.w = self.nu*2*np.pi
        self.Tw = Tw

    def mole_heat_capacity(self,T,Natom):
        nu = self.nu
        pdos = self.pdos
        x = C.Planck*nu/(kB*T) 
        pdos = pdos / (np.trapz(pdos,x))   # normalzed pDOS
        C_w = kB * x**2 * np.exp(x) / (np.exp(x) - 1.0)**2
        return 3 * Natom * NA * np.trapz(C_w*pdos, x) # J/(mol K)

    def set_nu(self, nu):
        '''
        nu: w/2pi in THz
        '''
        nu = nu*1.0e12
        if hasattr(self, 'pdos'):
            self.pdos = np.interp(nu, self.nu, self.pdos)
        if hasattr(self,'Tw'):
            self.Tw = np.interp(nu, self.nu, self.Tw)
        self.nu_THz = nu
        self.nu = nu
        self.w = nu*2*np.pi
        self.w_THz = self.w / 1.0e12
    
    def set_mfp(self, mfp):
        self.mfp = mfp/1.0e9
    
    def get_mfp(self):
        if hasattr(self, 'mfp'):
            return self.mfp
        elif hasattr(self, 'mfp_factors'):
            A,n = self.mfp_factors
            self.mfp = mfp_power(self.w_THz, A, n)
            return self.mfp

    @property
    def debye_temp(self):
        v = self.v
        natom = self.natom
        return v * (hbar/kB)*(6 * (np.pi)**2 * natom)**(1/3)

    @property
    def debye_freq(self):
        return kB*self.debye_temp/C.Planck
    
    @property
    def copy(self):
        import copy
        return copy.deepcopy(self)
    
def mfp_power(w, A, n, lam_min=0.0, lam_max=1.0e5):
    '''
    lambda = A*w^(-n)
    A: float
    w: in THz
    n: positive integer
    '''
    lam = A*w**(-n)*1.0e-9 
    lam[np.where(lam<lam_min*1.0e-9)] = lam_min*1.0e-9
    lam[np.where(lam>lam_max*1.0e-9)] = lam_max*1.0e-9
    return lam

def g_factor(x):
    def fit_fun(x):
        # use pre-fitted parameter to calcute fastly
        return(2.37341255*x**0.87796735)
    if x < 1e-4:
        return fit_fun(x)
    else:
        power = np.linspace(-10, 0, 10000)
        mu = 10**power
        E3 = np.trapz(np.power(mu,1)*np.exp(-x/mu),mu)
        E5 = np.trapz(np.power(mu,3)*np.exp(-x/mu),mu)
        return 1-3/8*(1 - 4*(E3 - E5))/x

def _root_mean_squre_deviation(test_data, ref_data):
    test_data = np.array(test_data)
    ref_data = np.array(ref_data)
    n = len(test_data)
    return np.sqrt(np.sum( ((test_data-ref_data)/ref_data)**2 ) / n)

class Calculator():
    def __init__(self, material, nu_min, nu_max, nu_cut, dnu, T):
        #self.nu_THz = np.linspace(nu_min, nu_max, int((nu_max-nu_min)/dnu))
        self.set_nu(nu_min, nu_max, dnu)
        self.w_THz = 2*np.pi*self.nu_THz
        self.nu = self.nu_THz * 1.0e12
        self.w =  2*np.pi*self.nu
        self.nu_cutoff = nu_cut*1.0e12
        self.T = np.array(T)
        self.material = copy.deepcopy(material)
        
    def set_nu(self,nu_min, nu_max, dnu):
        n_nu = int((nu_max-nu_min)/dnu)
        power = np.linspace(np.log2(nu_min), np.log2(nu_max), n_nu)
        self.nu_THz = 2.0**power


class Minimal(Calculator):
    def __init__(self, material, nu_min=0.01, nu_max=100.0, nu_cut=20.0, T=[300.0], dnu=0.01):
        super(Minimal,self).__init__(material, nu_min, nu_max, nu_cut, dnu, T)

    def diffuson(self):
        self.material.pdos = np.interp(self.nu, self.material.nu, self.material.pdos)
        TT = self.T
        natom = self.material.natom
        pdos = self.material.pdos

        self.kappa = np.zeros(len(TT))
        for i,T in enumerate(TT):
            x = C.Planck*self.nu/(kB*T)
            x_cut = C.Planck*self.nu_cutoff/(kB*T)
            pdos = self._normalize(x, pdos)
            kernel = pdos*x*(kB * x**2 * np.exp(x) / (np.exp(x) - 1.0)**2)    
            self.kappa[i] = natom**(1/3)/np.pi * kB*T/hbar * np.trapz(kernel[np.where(x<x_cut)],x[np.where(x<x_cut)])

        return self.kappa

    def diffuson_high_T(self):
        natom = self.material.natom
        v = self.material.v
        return 0.76*natom**(2/3) * kB * v

    def phonon_glass(self):
        v = self.material.v
        pass

    def phonon_glass_high_T(self):
        natom = self.material.natom
        v = self.material.v
        return 1.21*natom**(2/3) * kB * v

    def _normalize(self,x,y):
        return y / np.trapz(y,x)

class SingleMaterCalculator(Calculator):
    def __init__(self, 
                 material: Material, 
                 L: np.ndarray=[1.0e3], 
                 d: np.ndarray=1e5, 
                 nu_min: float=0.01,  # truncate the ultra low frequencies
                 nu_max: float=100.0, 
                 nu_cut: float=20.0, 
                 T: float=[300.0], 
                 dnu: float=0.01):
        super(SingleMaterCalculator,self).__init__(material, nu_min, nu_max, nu_cut, dnu, T)
        self.L = np.array(L)/1.0e9
        self.d = np.array(d)/1.0e9
        self.xi_ref = np.power(10, np.arange(-4,5,0.05))
        self.g_ref = np.array([g_factor(x) for x in self.xi_ref])
        self.material.set_nu(self.nu_THz)
        
    def bulk_classical(self): 
        nL = self.L.shape[0]
        self.kappa = np.zeros(nL)
        for i,L in enumerate(self.L):
            kernel = self.Tw/(1+L/self.mfp)
            mask = np.where(self.nu < self.nu_cutoff)
            self.kappa[i] = L*kB*np.trapz(kernel[mask],self.nu[mask])
        return self.kappa

    def bulk_quantum(self):
        nT = self.T.shape[0]
        nL = self.L.shape[0]
        self.kappa=np.zeros((nT,nL))

        for i, T in enumerate(self.T):
            pfpt = lambda x: (x/T)*np.exp(x)*(np.exp(x)-1)**(-2)
            for j, L in enumerate(self.L):
                x = C.Planck*self.nu/(kB*T)
                kernel = self.Tw/(1+L/self.mfp)*(C.Planck*self.nu)*pfpt(x)
                mask = np.where(self.nu < self.nu_cutoff)
                self.kappa[i,j] = L*np.trapz(kernel[mask], self.nu[mask])
        return self.kappa
    
    def fit_mfp_factors(self, L_ref, kappa_ref, A_range, n_range, npoints=101, quantum=False):
        L_ref = np.array(L_ref)
        kappa_ref = np.array(kappa_ref)
        rmsd = np.zeros((npoints,npoints))
        A = np.linspace(A_range[0], A_range[1], npoints)
        n = np.linspace(n_range[0], n_range[1], npoints)
        for i in range(npoints):
            for j in range(npoints):
                test_calculator = self.copy
                test_calculator.L = L_ref/1.0e9
                test_calculator.material.mfp_factors = [A[i], n[j]]
                if not quantum:
                    kappa_test = test_calculator.bulk_classical()
                else:
                    kappa_test = test_calculator.bulk_quantum()[0]
                rmsd[i,j] = _root_mean_squre_deviation(kappa_test, kappa_ref)
        
        return A, n, rmsd

    def film_classical(self):
        nL = self.L.shape[0]
        nd = self.d.shape[0]
        self.kappa = np.zeros((nL,nd))
        for i,L in enumerate(self.L):
            for j,d in enumerate(self.d):
                g = np.interp(d/self.mfp, self.xi_ref, self.g_ref)
                kernel = self.Tw/(1+L/self.mfp)*g
                mask = np.where(self.nu < self.nu_cutoff)
                self.kappa[i,j] = L*kB*np.trapz(kernel[mask],self.nu[mask])
        return self.kappa
    
    def get_film_classical(self, L, d):
        L = L/1.0e9
        d = d/1.0e9
        g = np.interp(d/self.mfp, self.xi_ref, self.g_ref)
        kernel = self.Tw/(1+L/self.mfp)*g
        mask = np.where(self.nu < self.nu_cutoff)
        kappa = L*kB*np.trapz(kernel[mask],self.nu[mask])
        return kappa

    def film_quantum(self):
        nT = self.T.shape[0]
        nL = self.L.shape[0]
        nd = self.d.shape[0]
        self.kappa = np.zeros((nT,nL,nd))
        for i,T in enumerate(self.T):
            pfpt = lambda x: (x/T)*np.exp(x)*(np.exp(x)-1)**(-2)
            for j,L in enumerate(self.L):
                for k,d in enumerate(self.d):
                    x = C.Planck*self.nu/(kB*T)
                    g = np.interp(d/self.mfp, self.xi_ref, self.g_ref)
                    kernel = self.Tw/(1+L/self.mfp)*(C.Planck*self.nu)*pfpt(x)*g
                    mask = np.where(self.nu < self.nu_cutoff)
                    self.kappa[i,j,k] = L*np.trapz(kernel[mask],self.nu[mask])
        return self.kappa

    def film_gray(self):
        kappa_max = self.material.kappa_max
        nL = self.L.shape[0]
        nd = self.d.shape[0]
        self.kappa = np.zeros((nL,nd))
        for i,L in enumerate(self.L):
            for j,d in enumerate(self.d):
                g = np.interp(d/self.mfp, self.xi_ref, self.g_ref)
                self.kappa[i,j] = kappa_max/(1+self.mfp/L)*g
        return self.kappa
    
    def mfp_spectra(self, L, d=1.0e8):
        d = d/1.0e9
        L = L/1.0e9
        mask_nu_cut = np.where(self.nu < self.nu_cutoff)
        mfp = self.mfp
        g = np.interp(d/mfp, self.xi_ref, self.g_ref)
        kernel = self.Tw/(1+L/self.mfp)*g
        nu = self.nu
       
        mfp_spectra = np.zeros(len(mfp))
        for i,l in enumerate(mfp):
            mask_mfp = np.ones(len(mfp))
            mask_mfp[mfp>l] = 0.0
            mfp_spectra[i] = np.trapz(kernel[mask_nu_cut]*mask_mfp[mask_nu_cut], nu[mask_nu_cut])
        order = np.argsort(mfp)
        mfp = np.sort(mfp)
        mfp_spectra = mfp_spectra[order]
        kappa = mfp_spectra[-1]
        mfp_spectra = mfp_spectra/mfp_spectra[-1]
        return mfp*1.0e9, mfp_spectra

    def accumulate_thermal_conductivity(self, L, d=1.0e8):
        d = d/1.0e9
        L = L/1.0e9
        mask_nu_cut = np.where(self.nu < self.nu_cutoff)
        g = np.interp(d/self.mfp, self.xi_ref, self.g_ref)
        kernel = self.Tw/(1+L/self.mfp)*g
        nu = self.nu
       
        accu_kappa = np.zeros(len(nu))
        for i,n in enumerate(nu):
            mask_nu = np.ones(len(nu))
            mask_nu[nu>n] = 0.0
            accu_kappa[i] = np.trapz(kernel[mask_nu_cut]*mask_nu[mask_nu_cut], nu[mask_nu_cut])
        kappa = accu_kappa[-1]
        accu_kappa = accu_kappa/accu_kappa[-1]
        return accu_kappa, kappa

    @property
    def bulk_max(self):
        g = 1.0
        kernel = self.Tw*self.mfp*g
        mask = np.where(self.nu < self.nu_cutoff)
        return kB*np.trapz(kernel[mask],self.nu[mask])
    
    @property
    def effective_mfp_factor(self):
        w = self.w_THz
        Tw = self.Tw
        kappa_max = self.material.kappa_max
        mask = np.where(self.nu < self.nu_cutoff)
        kernel = kB * Tw * w**(-2) * 1e-9
        denominator = np.trapz(kernel[mask], self.nu[mask])
        return kappa_max / denominator

    @property
    def copy(self):
        return copy.deepcopy(self)
        
    @property
    def mfp(self):
        return self.material.get_mfp()
    
    @property
    def Tw(self):
        return self.material.Tw
        
class SuperlattceCalculator(Calculator):
    def __init__(self,
                 material: Material, 
                 L: np.ndarray, 
                 d: np.ndarray,
                 d_ratio: np.ndarray,
                 nu_min: float=0.01, 
                 nu_max: float=100.0, 
                 nu_cut: float=20.0, 
                 T: float=[300.0], 
                 dnu: float=0.01):
        super(SuperlattceCalculator,self).__init__(material, nu_min, nu_max, nu_cut, dnu, T)
        self.nu_cutoff = nu_cut
        self.d_ratio = np.array(d_ratio)
        self.L = np.array(L)
        self.d = np.array(d)

        
    def classical(self):
        self.kappa = []
        nd = self.d.shape[0]
        self.kappa_avg = np.zeros(nd)
        for i, mater in enumerate(self.material):
            calculator = SingleMaterCalculator(mater, L=[self.L[i]], d=self.d*self.d_ratio[i], nu_cut=self.nu_cutoff)
            kappa = calculator.film_classical()[0]
            self.kappa.append(kappa)
            self.kappa_avg += self.d_ratio[i]*kappa
        return self.kappa_avg, self.kappa
    
    def classical_variable_L(self):
        self.kappa = []
        nd = self.d.shape[0]
        self.kappa_avg = np.zeros(nd)
        for i, mater in enumerate(self.material):
            dd = self.d*self.d_ratio[i]
            calculator = SingleMaterCalculator(mater, L=[self.L[i]], d=dd, nu_cut=self.nu_cutoff)
            kappa = np.zeros(nd)
            for j,d in enumerate(dd):
                calculator.L = np.array([min(self.L[i],d)])/1.0e9
                calculator.d = np.array([min(self.L[i],d)])/1.0e9
                kappa[j] = calculator.film_classical()[0,0]
            self.kappa.append(kappa)
            self.kappa_avg += self.d_ratio[i]*kappa
        return self.kappa_avg, self.kappa
    
    def gray(self):
        self.kappa = []
        nd = self.d.shape[0]
        self.kappa_avg = np.zeros(nd)
        for i, mater in enumerate(self.material):
            calculator = SingleMaterCalculator(mater, L=[self.L[i]], d=self.d*self.d_ratio[i], nu_cut=self.nu_cutoff)
            kappa = calculator.film_gray()[0]
            self.kappa.append(kappa)
            self.kappa_avg += self.d_ratio[i]*kappa
        return self.kappa_avg, self.kappa
    
    def gray_variable_L(self):
        self.kappa = []
        nd = self.d.shape[0]
        self.kappa_avg = np.zeros(nd)
        for i, mater in enumerate(self.material):
            dd = self.d*self.d_ratio[i]
            calculator = SingleMaterCalculator(mater, L=[self.L[i]], d=dd, nu_cut=self.nu_cutoff)
            kappa = np.zeros(nd)
            for j,d in enumerate(dd):
                calculator.L = np.array([min(self.L[i],d)])/1.0e9
                calculator.d = np.array([min(self.L[i],d)])/1.0e9
                
                kappa[j] = calculator.film_gray()[0,0]
            self.kappa.append(kappa)
            self.kappa_avg += self.d_ratio[i]*kappa
        return self.kappa_avg, self.kappa
    
        
    @property
    def film_max(self):
        pass
    
    
    def quantum(self):
        "TODO: fix this function, if usable"
        pass
    

if __name__ == '__main__':

    pass
