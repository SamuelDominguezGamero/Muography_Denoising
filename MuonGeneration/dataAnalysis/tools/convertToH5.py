import optparse
import pandas as pd
import ROOT as r
import numpy as np

if __name__=='__main__':

    parser = optparse.OptionParser(usage='usage: %prog [options] path', version='%prog 1.0')
    parser.add_option('-i', '--input', action='store', type='string', dest='input', default='input.root', help='Input file')
    parser.add_option('-o', '--output', action='store', type='string', dest='output', default='output.h5', help='Output file')

    (opts,args) = parser.parse_args()

    f = r.TFile(opts.input)
    t = f.Get('events')

    data = []
    
    for ev in t:
        data.append([ev.x1, ev.y1, ev.z1, ev.vx1, ev.vy1, ev.vz1, ev.x2, ev.y2, ev.z2, ev.vx2, ev.vy2, ev.vz2])
        if len(data) % 100000 == 0:
            print(f"[INFO] Processed {len(data)} / {t.GetEntries()} events...")
            print(f"[INFO] Completed {len(data)/t.GetEntries()*100:.2f}% of events")
            print(60 * "=")
    npdata = np.asarray(data)
    df = pd.DataFrame(npdata, columns=['x1', 'y1', 'z1', 'vx1', 'vy1', 'vz1', 'x2', 'y2', 'z2', 'vx2', 'vy2', 'vz2'])
    df.to_hdf(opts.output, key='df', mode='w')