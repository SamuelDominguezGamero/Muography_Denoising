import json, sys, optparse
import ROOT as r
from array import array
import sys
from tools.Event import Event


class EventLoader:

    def __init__(self, inputFile, configuration):
        self.inputFile = inputFile
        self.configuration = configuration
             
    def loadEvents(self):

        try:
            with open(self.configuration, 'r') as cinput:
                data = json.load(cinput)
        except:
            print('Configuration file is not valid')
            sys.exit()
        cinput.close()

        nDet = len(data['Detectors'])
        nLayer = len(data['Detectors'][0]['Layers'])
    
        try:
            f = r.TFile(self.inputFile)
        except:
            print('Input file does not exist or it is corrupt')
            sys.exit()

        events = []
        event = -1
        counter = -1

        for ev in f.hits:
            if event != ev.eventNumber:
                newEvent = Event(ev.eventNumber, nDet, nLayer)
                newEvent.add(ev.det, ev.layer, ev.energy, ev.x, ev.y, ev.z, ev.vx, ev.vy, ev.vz, ev.localx, ev.localy, ev.localz, ev.localvx, ev.localvy, ev.localvz, ev.genID)
                events.append(newEvent)
                event = ev.eventNumber
                counter = counter + 1
            else:
                events[counter].add(ev.det, ev.layer, ev.energy, ev.x, ev.y, ev.z, ev.vx, ev.vy, ev.vz, ev.localx, ev.localy, ev.localz, ev.localvx, ev.localvy, ev.localvz, ev.genID)

        f.Close()
    
        return events


 
        
