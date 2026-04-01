import optparse
import pandas as pd
import ROOT as r
import numpy as np

if __name__=='__main__':

    parser = optparse.OptionParser(usage='usage: %prog [options] path', version='%prog 1.0')
    parser.add_option('-i', '--input', action='store', type='string', dest='input', default='input.root', help='Input file')
    parser.add_option('-o', '--output', action='store', type='string', dest='output', default='output.root', help='Output file')

    (opts,args) = parser.parse_args()

    f = r.TFile(opts.input)
    t = f.Get('events')

    data = []
    
    for ev in t:
        data.append([ev.x, ev.y, ev.z, ev.vx, ev.vy, ev.vz])

    npdata = np.asarray(data)
    df = pd.DataFrame(npdata, columns=['x', 'y', 'z', 'vx', 'vy', 'vz'])
    df.to_hdf(opts.output, key='df', mode='w')




