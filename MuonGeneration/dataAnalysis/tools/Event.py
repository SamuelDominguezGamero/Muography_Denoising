import json, sys, optparse
import ROOT as r
from array import array

class Event:

    def __init__(self, nEvent, ndet, nlayer):
        self.nEvent = nEvent
        self.ndet = ndet
        self.nlayer = nlayer
        self.det = []
        self.layer = []
        self.Energy = []
        self.localx = []
        self.localy = []
        self.localz = []
        self.x = []
        self.y = []
        self.z = []
        self.localvx = []
        self.localvy = []
        self.localvz = []
        self.vx = []
        self.vy = []
        self.vz = []
        self.genID = []

        for i in range(ndet):
            self.det.append([])
            self.layer.append([])
            self.Energy.append([])
            self.localx.append([])
            self.localy.append([])
            self.localz.append([])
            self.x.append([])
            self.y.append([])
            self.z.append([])
            self.localvx.append([])
            self.localvy.append([])
            self.localvz.append([])
            self.vx.append([])
            self.vy.append([])
            self.vz.append([])
            self.genID.append([])


    def add(self, det_, layer_, Energy_, x_, y_, z_, vx_, vy_, vz_, localx_, localy_, localz_, localvx_, localvy_, localvz_, genID_):
       
        if det_ in self.det[det_] and layer_ in self.layer[det_] or abs(genID_) != 13:
            return  
        self.det[det_].append(det_)
        self.layer[det_].append(layer_)
        self.Energy[det_].append(Energy_)
        self.localx[det_].append(localx_)
        self.localy[det_].append(localy_)
        self.localz[det_].append(localz_)
        self.localvx[det_].append(localvx_)
        self.localvy[det_].append(localvy_)
        self.localvz[det_].append(localvz_)
        self.x[det_].append(x_)
        self.y[det_].append(y_)
        self.z[det_].append(z_)
        self.vx[det_].append(vx_)
        self.vy[det_].append(vy_)
        self.vz[det_].append(vz_)
        self.genID[det_].append(genID_)

    def nHits(self):
        n = 0
        for i in range(self.ndet):
            n = n + len(self.det[i])
        return n
   

    def Print(self):
        print('-------------------New----------------')
        for i in range(self.ndet):
            for h in range(len(self.det[i])):       
                print(self.det[i][h], self.layer[i][h], self.x[i][h], self.y[i][h], self.z[i][h])


    def makeFit(self, j):

        x = y = z = xz = yz = zz = 0
        n = len(self.x[j])
        for i in range(n):
            x += self.x[j][i]
            y += self.y[j][i]
            z += self.z[j][i]
            xz += self.x[j][i]*self.z[j][i]
            yz += self.y[j][i]*self.z[j][i]
            zz += self.z[j][i]*self.z[j][i]
        dxdz = (n * xz - x * z)/(n * zz - z*z)
        x0 = x/n - dxdz * z/n
        dydz = (n * yz - y * z)/(n * zz - z*z)
        y0 = y/n - dydz * z/n
        z0 = z/n
        return x0, y0, z0, dxdz, dydz, -1.0, self.Energy[j][0]
 
    def validEvent(self):

        if self.nHits() < self.nlayer*self.ndet:
            return False
        
        lista = [i for i in range(0, self.ndet*self.nlayer)]
        for d in range(self.ndet):
            for i in range(len(self.det[d])):
                id = self.det[d][i] * self.nlayer + self.layer[d][i]
                if id in lista:
                    lista.remove(id)
        if len(lista) != 0:
            return False
        return True
    
  
