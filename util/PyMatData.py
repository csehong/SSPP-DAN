# =========================================
# PyMatData.py
# Author: Woobin Im
# Description: Reading MatConvNet
#              model & make it clean
# =========================================
import numpy as np
import scipy.io as sio



class PyMatData:
    def __init__(self, matpath):
        try:
            self.matnet = sio.loadmat(matpath)
        except IOError as e:
            print('IO Error: {0}'.format(e))
        except Exception as e:
            print("Unexpected error: {0}".format(e))

        try:
            self.matlayers = self.matnet['layers'][0]
        except KeyError as e:
            print('Key Error in mat file: {0}'.format(e))
        except Exception as e:
            print("Unexpected error: {0}".format(e))

        if 'meta' in self.matnet.keys():
            self.matmeta = self.matnet['meta']
        self.layers = []
        self.meta = {}
        self.loadnet()

    def printnet(self):
        for l in range(len(self.matlayers)):
            layer = self.matlayers[l]
            attrs = layer.dtype.names
            print 'Layer_%d {' % (l)
            for attr in attrs:
                attrval = layer[attr][0, 0]
                if attr == 'weights':
                    if len(layer[attr][0, 0]) == 0:
                        continue
                    print ' %s: %s,%s' % (attr, layer[attr][0, 0][0, 0].shape, layer[attr][0, 0][0, 1].shape)
                    continue
                if len(attrval) > 0:
                    print ' %s: %s' % (attr, attrval[0])
            print '}'

    def loadnet(self):
        for l in range(len(self.matlayers)):
            layer = self.matlayers[l]
            attrs = layer.dtype.names
            toadd = {}
            for attr in attrs:
                attrval = layer[attr][0, 0]
                if attr == 'weights':
                    if len(layer[attr][0, 0]) == 0:
                        continue
                    toadd[attr] = [layer[attr][0, 0][0, 0], layer[attr][0, 0][0, 1][:, 0]]
                    continue
                if len(attrval) > 0:
                    toadd[attr] = attrval[0]
            self.layers.append(toadd)
        if 'meta' in self.matnet.keys():
            self.meta = self.StructuredArray2Dict(self.matmeta)

    def StructuredArray2Dict(self, arr):
        ret = {}
        try:
            keys = arr.dtype.names
        except:
            pass
        if keys is None:
            return np.squeeze(arr)
        for key in keys:
            if arr[key].size == 1:
                n_val = arr[key][0, 0]
            else:
                n_val = arr[key]
            ret[key] = n_val
            ret[key] = self.StructuredArray2Dict(n_val)
        return ret