'''
Created on Feb 24, 2012

@author: Jacob Frelinger
'''


def text_size(dict,delim):
    rslt = delim
    for i in dict:
        rslt += '$%s%s%s%s' % (i, delim, dict[i], delim)
    size = len(rslt)
    return size, rslt

def export_fcs(name, pnts, channels, extra=None):
    # magic fcs defined positions
    header_text_start = (10, 17)
    header_text_end = (18, 25)
    header_data_start = (26, 33)
    header_data_end = (34, 41)
    header_analysis_start = (42, 49)
    header_analysis_end = (50, 5)
    
    
    fh = open(name,'wb')
    fh.write('FCS3.1')
    fh.write(' '*53)
    
    
    ## WRITE TEXT Segment
    text_start = 256 # arbitrarilly start at byte 256.
    delim = '/' # use / as our delimiter.
    #write spaces untill the start of the txt segment
    fh.seek(58)
    fh.write(' '*(text_start-fh.tell()))
    
    nchannels = pnts.shape[1]
    npnts = pnts.shape[0]
    datasize = 4*nchannels*npnts # 4 is bytes to hold float
    
    text = {}
    text['BEGINANALYSIS'] = '0'
    text['BEGINDATA'] = '0'
    text['BEGINSTEXT'] = '0'
    text['BYTEORD'] = '1,2,3,4' # little endian
    text['DATATYPE'] = 'F' # only do float data
    text['ENDANALYSIS'] = '0'
    text['ENDDATA'] = '0'
    text['ENDSTEXT'] = '0'
    text['MODE'] = 'L' # only do list mode data
    text['NEXTDATA'] = '0'
    text['PAR'] = str(nchannels)
    text['TOT'] = str(npnts)
    for i in range(nchannels):
        text['P%dB' % (i+1)] = '32' # datatype =f requires 32 bits
        text['P%dE' % (i+1)] = '0,0'
        #text['P%dR' % (i+1)] = '2623144'
        text['P%dR' % (i+1)] = str(int(pnts[:,i].max()))
        text['P%dN' % (i+1)] = channels[i]
        
    if extra is not None:
        for i in extra:
            i = i.strip()
            if i.lower() not in text and i.upper() not in text:
                text[i] = extra[i]
        
    i = 1
    size, _ =text_size(text, delim)
    prop_size = text_start+((size%256)+i) * 256
    text['BEGINDATA'] = prop_size
    text['ENDDATA'] = prop_size+datasize
    data_start = prop_size
    data_end = prop_size+datasize-1
    size, text_segment = text_size(text,delim)
    text_end = text_start+size-1
    
    fh.write(text_segment)
       
    
    #print fh.tell(), text_end
    fh.write(' '*(data_start-fh.tell()))

    fh.write(pnts.astype('<f').tostring())
    
    fh.seek(header_text_start[0])
    fh.write(str(text_start))
    fh.seek(header_text_end[0])
    fh.write(str(text_end))
    
    fh.seek(header_data_start[0])
    if len(str(data_end))< (header_data_end[1]-header_data_end[0]):
        fh.write(str(data_start))
        fh.seek(header_data_end[0])
        fh.write(str(data_end))
    else:
        fh.write(str(0))
        fh.seek(header_data_end[0])
        fh.write(str(0))
    
    fh.seek(header_analysis_start[0])
    fh.write(str(0))
    fh.seek(header_analysis_end[0])
    fh.write(str(0))
    
    
    fh.close()
    
