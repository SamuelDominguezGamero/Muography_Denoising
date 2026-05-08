import json, sys, optparse
import ROOT as r
from array import array
from tools.Event import Event
from tools.EventLoader import EventLoader

print("[INFO] ----- Libraries and tools successfully loaded, starting track correlations")

if __name__ == '__main__':

    parser = optparse.OptionParser(usage='usage: %prog [options] path', version='%prog 1.0')
    parser.add_option('-i', '--input', action='store', type='string', dest='inputFile', default='input.root', help='Input ROOT File')
    parser.add_option('-c', '--conf', action='store', type='string', dest='configurationFile', default='conf.json', help='Configuration file')
    parser.add_option('-o', '--output', action='store', type='string', dest='outputFile', default='output.root', help='Output ROOT file')
    (opts, args) = parser.parse_args()
 
    loader = EventLoader(opts.inputFile, opts.configurationFile)
    events = loader.loadEvents()

    print('Number of events', len(events))
    
    try:
        output = r.TFile(opts.outputFile, 'RECREATE')
    except:
        print('Cannot open output file')
        sys.exit()


    #Preparing the structure for the output tree
    t = r.TTree('events', 'events')
    nevent = array('i', [0])
    x1 = array('f', [0])
    y1 = array('f', [0])
    z1 = array('f', [0])
    vx1 = array('f', [0])
    vy1 = array('f', [0])
    vz1 = array('f', [0])
    energy1 = array('f', [0])
    x2 = array('f', [0])
    y2 = array('f', [0])
    z2 = array('f', [0])
    vx2 = array('f', [0])
    vy2 = array('f', [0])
    vz2 = array('f', [0])
    energy2 = array('f', [0])
    
    t.Branch('nevent', nevent, 'nevent/I')
    t.Branch('x1', x1, 'x1/F')
    t.Branch('y1', y1, 'y1/F')
    t.Branch('z1', z1, 'z1/F')
    t.Branch('vx1', vx1, 'vx1/F')
    t.Branch('vy1', vy1, 'vy1/F')
    t.Branch('vz1', vz1, 'vz1/F')
    t.Branch('energy1', energy1, 'energy1/F')
    t.Branch('x2', x2, 'x2/F')
    t.Branch('y2', y2, 'y2/F')
    t.Branch('z2', z2, 'z2/F')
    t.Branch('vx2', vx2, 'vx2/F')
    t.Branch('vy2', vy2, 'vy2/F')
    t.Branch('vz2', vz2, 'vz2/F')
    t.Branch('energy2', energy2, 'energy2/F')
   
    counter = 0
    valid   = 0
    invalid = 0
    print("Starting loop over events")
    for ev in events:       
        counter = counter + 1
        if not ev.validEvent():
            continue    
        x1_, y1_, z1_, vx1_, vy1_, vz1_, energy1_ = ev.makeFit(0)
        x2_, y2_, z2_, vx2_, vy2_, vz2_, energy2_ = ev.makeFit(1)
        
        if not ev.validEvent():
            invalid += 1
            continue
        valid += 1
        nevent[0] = ev.nEvent
        # print(ev.nEvent)
        # ev.Print()
        x1[0] = x1_
        y1[0] = y1_
        z1[0] = z1_
        vx1[0] = vx1_
        vy1[0] = vy1_
        vz1[0] = vz1_
        energy1[0] = energy1_
        x2[0] = x2_
        y2[0] = y2_
        z2[0] = z2_
        vx2[0] = vx2_
        vy2[0] = vy2_
        vz2[0] = vz2_
        energy2[0] = energy2_
        t.Fill()
    print("[FINISHED] ----- Loop over events finished")
    print("\n")
    print("----------------------------------------------------------------")
    print(f"{counter} events has been processed trough the script makeHLTuple.py")
    print(f"Valid events: {valid}, Invalid events: {invalid}")
    print("----------------------------------------------------------------")
    print("\n")

    output.Write()
    output.Close()
