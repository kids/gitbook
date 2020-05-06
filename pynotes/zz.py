
# style, display etc.
def ipy_width(w='90%'):
    import IPython
    IPython.core.display.display(IPython.core.display.HTML('<style>.container { width:'+w+' !important; }</style>'))

def plotly_layout():
    return go.Layout(
        barmode='stack',
        paper_bgcolor='rgb(199,237,204)',
        plot_bgcolor='rgb(199,237,204)',
        xaxis=dict(
                  type='category',
        ),)

# parallel
def parallel_serv_start():
    import subprocess
    subprocess.Popen(['ipcluster','start'])
    import ipyparallel as ipp
    rc = ipp.Client()
    # ret =rc[:2].apply_async(frontslope_err,(p,3))
    # ret =rc[:2].imap(frontslope_err,[(p,3) for p in [[1.,2.,4.,7.],[4.,2.,5.,2.]]])
    # for _ in mtqdm(pool.imap_unordered(do_work, tasks), total=len(tasks)):
    # rc.close()

# losses
def tf_quantile_loss(y,output):
    import tensorflow as tf
    error = tf.subtract(y, output)
    loss = tf.reduce_mean(tf.maximum(q*error, (q-1)*error), axis=-1)

def torch_quantile_loss():
    import torch
    class QuantileLoss(torch.nn.Module):
        def __init__(self, quantiles):
            super().__init__()
            self.quantiles = quantiles
            
        def forward(self, preds, target):
            assert not target.requires_grad
            assert preds.size(0) == target.size(0)
            losses = []
            for i, q in enumerate(self.quantiles):
                errors = target - preds[:, i]
                losses.append(
                    torch.max(
                       (q-1) * errors, 
                       q * errors
                ).unsqueeze(1))
            loss = torch.mean(
                torch.sum(torch.cat(losses, dim=1), dim=1))
            return loss
    return QuantileLoss


# fetch data
def get_future_data(fl='/data/MA/',sample_freq='5Min'):
    import pandas as pd
    from tqdm.notebook import tqdm
    dt=pd.DataFrame()
    for day in tqdm(sorted(os.listdir(fl))[61:90]):
        try:
            fls=os.listdir(fl+day)
            tdt=[pd.read_csv(fl+day+'/'+fj)[['date','time','latest_price']] for fj in fls]
            tdt=max(tdt,key=lambda k:k.shape[0])
        except:
            print(day)
            continue
        dtm=tdt['date'].astype(str)+' '+(tdt['time']/1000).astype(int).astype('str').str.zfill(6)
        tdt.index=pd.to_datetime(dtm, format='%Y%m%d %H%M%S')
        tdt=tdt.drop(columns=['date','time'])
        tdt=tdt.resample('1Min').ohlc()['latest_price']['open'].dropna()
        tdt.columns =['time','close']
        dt=pd.concat([dt,tdt])
    dt=dt.resample(sample_freq).ohlc()[0].dropna()
    return dt


# feature generators
def feat_fft(seq, skip=1, win_size=60, fft_size=7):
    import pandas as pd
    if isinstance(seq,pd.Series):
        z=seq.tolist()
    seq_list = [seq[i+1-win_size*skip:i+1:skip] for i in range(len(seq))]
    x_feat = [ np.fft.fft(seq)[1:fft_size] if len(seq)>0
                  else [0+0j]*(fft_size-1)
                    for seq in seq_list ]
    x_feat = [ np.hstack([[j.real,j.imag]
                               for j in i]).tolist()
                    for i in x_feat ]
    return pd.Series(x_feat)

def feat_1h(x):
    import numpy as np
    v=x[::120].copy()
#     v['l6']=(v.Last.rolling(6).max().shift(-6)-v.Last*2
#                 +v.Last.rolling(6).min().shift(-6))
    v['label_r10'] = (v.price-v.price.shift(-1))/v.price.shift(-1)
#     v['label_r20'] = qcut(v.l6).cat.codes

    v['log']=np.log(v.price)
    v['turn_log']=np.log(v.amount)
    
    f1=x_df([0]*(60*1-1)+v.log.tolist())[60*1-1:]
    for i in range(8):
        v['f1_'+str(i)]=[j[i]*100 for j in f1]
        v['f1_'+str(i)][v['f1_'+str(i)].abs()>3]=0
    f6=x_df([0]*(60*6-1)+v.log.tolist(),skip=6)[60*6-1:]
    for i in range(8):
        v['f6_'+str(i)]=[j[i]*0.5 for j in f6]
        v['f6_'+str(i)][v['f6_'+str(i)].abs()>3]=0
    
    fv1=x_df([0]*(60*1-1)+v.turn_log.tolist())[60*1-1:]
    for i in range(8):
        v['fv1_'+str(i)]=[j[i]*100 for j in fv1]
        v['fv1_'+str(i)][v['fv1_'+str(i)].abs()>3]=0
    fv6=x_df([0]*(60*6-1)+v.turn_log.tolist(),skip=6)[60*6-1:]
    for i in range(8):
        v['fv6_'+str(i)]=[j[i]*0.5 for j in fv6]
        v['fv6_'+str(i)][v['fv6_'+str(i)].abs()>3]=0

    thour=pd.get_dummies(v.t//10000000*10+v.t%10000000//3000000)
    for i in thour:
        v[i]=thour[i]
    v=v.replace([np.inf, -np.inf], np.nan)
    #v.fillna(0,inplace=True)
    v.dropna(inplace=True)
    y=v[[ 
               'TradingDay', 'label_r20',
               't', 'Last'
            ]]
    x=v[
            [ 'f1_'+str(i) for i in range(6) ]+
            [ 'f6_'+str(i) for i in range(6) ]+
            [ 'fv1_'+str(i) for i in range(6) ]+
            [ 'fv6_'+str(i) for i in range(6) ]+
            [91, 100, 101, 110, 130, 131, 140, 141]
        ]
    return y,x


# polyfit
def reconstruct(px,k=2):
    import numpy as np
    rec = np.polyval(np.polyfit(np.arange(len(px)),px,k),np.arange(len(px)))
    err = round(np.sum(np.abs(rec-px))/np.sum(px),4)
    return rec, err

def frontslope(px,k=2,i=0,j=0,m=1):
    import numpy as np
    if not (i==0 and j==0):
        i0=i if i-j<1 else i-j
        px=px[i0:i]
    diff = np.polyder(np.polyfit(np.arange(len(px)),px,k),m=m)
    rec = np.polyval(diff,np.arange(len(px)))
    slope = round(rec[-1],2)
    err = round(np.sum(np.abs(rec-px))/np.sum(px),4)
    if i==0 and j==0:
        return slope, err
    return (slope,err,i,j,k)

def conv_predict(px,k=2,n=1):
    import numpy as np
    return np.polyval(np.polyfit(np.arange(len(px)),px,k),np.arange(len(px)+n))[-1]

def reconstruct_plots(x,x0,x1,k=2,surname=''):
    import numpy as np
    tdt=x[x0:x1]
    return {'x':np.arange(x0,x1),'y':reconstruct(tdt,k),
            'name':'{}\t{}\t{}'.format(frontslope(tdt,k),err(tdt,k),surname)}

def get_mark_power(dt_close,iorder,jorder):
    import numpy as np
    from tqdm.notebook import tqdm
    k=[(i,i*10*2**j) for i in iorder for j in jorder]
    ok={j:i for i,j in enumerate(k)}
    nt=[[] for _ in k]
    nt2=[[] for _ in k]
    nt_err=[[] for _ in k]
    nt_fit=[[] for _ in jorder]
    for i in tqdm(range(n)):
        if i<10:
            for tnt in [nt,nt_err,nt_fit]:
                for j in tnt:
                    j.append(0)
            continue
        for j in jorder:
            tdiff,terr=-1e7,1e7
            for o in iorder:
                oj=o*10*2**j
                m=0 if i-oj<1 else i-oj
                jdiff,jerr = frontslope(dt_close[m:i],o)
                nt_err[ok[(o,oj)]].append(jerr)
                nt[ok[(o,oj)]].append(jdiff)
                if jerr<terr:
                    tdiff=jdiff
                    terr=jerr
            nt_fit[j].append(tdiff)
    nt=np.array(nt)
    nt_err=np.array(nt_err)
    nt_fit=np.array(nt_fit)
    return (k,nt,nt_err,nt_fit)

def get_mark_seq(dt_close,iorder,jorder):
    import numpy as np
    from tqdm.notebook import tqdm
    k=[(i,j) for i in iorder for j in jorder]
    ok={j:i for i,j in enumerate(k)}
    nt=[[] for _ in k]
    nt_err=[[] for _ in k]
    nt_fit=[[] for _ in jorder]
    for n in tqdm(range(len(dt_close))):
        if n<10:
            for tnt in [nt,nt_err,nt_fit]:
                for j in tnt:
                    j.append(0)
            continue
        for j in jorder:
            tdiff,terr=-1e7,1e7
            for i in iorder:
                jdiff,jerr,_,_,_=frontslope((dt_close,n,j,i))
                nt_err[ok[(i,j)]].append(jerr)
                nt[ok[(i,j)]].append(jdiff)
                if jerr<terr:
                    tdiff=jdiff
                    terr=jerr
            nt_fit[j-min(jorder)].append(tdiff)
    nt=np.array(nt)
    nt_err=np.array(nt_err)
    nt_fit=np.array(nt_fit)
    return (k,nt,nt_err,nt_fit)
    
def get_mark_seq_parallel(dt_close,iorder,jorder):
    # rc=ipp.Client()
    k=[(i,j) for i in iorder for j in jorder if i*10<=j]
    ok={j:i for i,j in enumerate(k)}
    nt=[[] for _ in k]
    nt2=[[] for _ in k]
    nt_err=[[] for _ in k]
    nt_fit=[[] for _ in jorder]
    for n in tqdm(range(len(dt_close))):
        if n<10:
            for tnt in [nt,nt_err,nt_fit]:
                for j in tnt:
                    j.append(0)
            continue
        jdata=dt_close.tolist()
        ret = rc[:7].imap(frontslope_err,[(jdata,n,j,i) for i in iorder for j in jorder if i*10<=j])
        for r in ret:
            _,j,i,jdiff,jerr = r
            nt_err[ok[(i,j)]].append(jerr)
            nt[ok[(i,j)]].append(jdiff)
    nt=np.array(nt)
    nt_err=np.array(nt_err)
    return (k,nt,nt_err)

# bt
def backtest(x,mark):
    import numpy as np
    assert len(x)==len(mark), f'lengths:{len(x)},{len(mark)}'
    pdata=[{'x':np.arange(len(x)),'y':x}]
    pmark=[]
    nmark=[]
    sign=0 if mark[0]==0 else int(2*(int(mark[0]>0)-0.5))
    lastpos, longgain, shortgain = 0,0,0
    for i in range(len(x)):
        if mark[i]>0 and sign<1:
            pmark.append(i)
            sign=1
            shortgain += 0 if lastpos==0 else lastpos-x[i]
            lastpos=x[i]
        if mark[i]<0 and sign>-1:
            nmark.append(i)
            sign=-1
            longgain += 0 if lastpos==0 else x[i]-lastpos
            lastpos=x[i]
    if sign==1 and lastpos!=0:
        nmark.append(len(x)-1)
        longgain+=x[-1]-lastpos
    elif sign==-1 and lastpos!=0:
        pmark.append(len(x)-1)
        shortgain+=lastpos-x[-1]
        
    pdata.append({'x':pmark,'y':[x[i] for i in pmark],'mode':'markers'})
    pdata.append({'x':nmark,'y':[x[i] for i in nmark],'mode':'markers'})
    gain = (longgain,shortgain,x[-1]-x[0])
    return (gain,pdata)


# RSRS
def rsrs(dt,n=16,m=300):
    import numpy as np
    dt['beta'] = 0
    dt['R2'] = 0
    for i in range(1,len(dt)-1):
        if i-n+1<2:
            continue
        df_ne=dt.iloc[ i-n+1:i]
        result=np.polyfit(df_ne['low'],df_ne['high'],1,full=True)
        dt.loc[dt.index[i+1],'beta'] = result[0][0]
        r2 = 1-result[1][0]/len(df_ne)/np.var(df_ne['high'],ddof=0)
        dt.loc[dt.index[i+1],'R2'] = 0 if r2<0 else r2
        #if 1-result[1][0]/len(df_ne)/np.var(df_ne['low'],ddof=0)<0:
        #    print(i-n+1,i,result,np.var(df_ne['low'],ddof=0))
        #    return(df_ne)

    dt['ret'] = dt.close.pct_change(1)
    dt['beta_norm'] = (dt['beta'] - dt.beta.rolling(m).mean().shift(1))/dt.beta.rolling(m).std().shift(1)
    # for i in range(m):
    #     dt.loc[dt.index[i],'beta_norm'] = (dt.loc[i,'beta'] - dt.loc[:i-1,'beta'].mean())/dt.loc[:i-1,'beta'].std()
    # dt.loc[2,'beta_norm'] = 0
    dt['RSRS_R2'] = dt.beta_norm*dt.R2
    dt = dt.fillna(0)
    dt['beta_right'] = dt.RSRS_R2*dt.beta
    return dt


# douban push comments
def get_douban_xml_rss():
    import requests
    from xml.etree import ElementTree
    headers = {'user-agent': 'my-app/0.0.1'}
    x=requests.get('https://www.douban.com/feed/review/movie',headers=headers,timeout=300)
    rt=ElementTree.fromstring(x.text)
    for i in rt[0]:
        if not i.tag=='item':
            continue
        t=i.find('description').text.split('\n')
        if t[2]=='评价: 力荐':
            print(t[1].split('评论: ')[1])


# cos
def cos_client():
    #!pip install cos-python-sdk-v5
    from qcloud_cos import CosConfig
    from qcloud_cos import CosS3Client
    secret_id = u''
    secret_key = u''
    region = u'ap-guangzhou'
    config = CosConfig(Secret_id=secret_id, Secret_key=secret_key, Region=region, Token=None)
    client = CosS3Client(config)

    # client.list_buckets()

    # client.upload_file(
    #     Bucket='xx-11111111',
    #     LocalFilePath='./t',
    #     Key='favi.ico'
    # )

    # client.list_objects(
    #     Bucket='xx-11111111',
    #     Prefix='fav'
    # )
    return client


# rl env
def rl_env():
    import pybullet as p
    import pybullet_data
    import time
    physicsClient = p.connect(p.DIRECT)#or p.GUI
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)
    planeId = p.loadURDF("plane.urdf")
    cubeStartPos = [0,0,1]
    cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
    boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)
    for i in range (10):
        p.stepSimulation()
        time.sleep(1./240.)
    cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    print(cubePos,cubeOrn)
    p.disconnect()


# inflexdb write
def write_influx():
    from influxdb import DataFrameClient
    client = DataFrameClient(host='localhost',port=8086)
    client.create_database('df')
    client.switch_database('df')
    test = [
            {
                "measurement": "m1",
                "tags": {"freq":"1Min"},
                "time": "2009-11-10T23:00:00Z",
                "fields": {
                    "mvalue1": 0.64,
                    "mvalue2": 3.12
                }},
            {
                "measurement": "m1",
                "tags": {"freq":"10Min"},
                "time": "2009-11-10T22:00:00Z",
                "fields": {
                    "mvalue1": 0.62,
                    "mvalue2": 3.11
                }},
            {
                "measurement": "m2",
                "tags": {"freq":"1Min"},
                "time": "2009-11-10T22:00:00Z",
                "fields": {
                    "mvalue1": 0.66,
                    "mvalue1": 3.11
                }}
            ]
    client.write_points(test)
    df = pd.DataFrame(data=list(range(30)),
                          index=pd.date_range(start='2014-11-16',
                                              periods=30, freq='H'), columns=['0'])
    client.write_points(df, 'df',
                            {'k1': 'v1', 'k2': 'v2'},protocol='line')


# online asr
def asr_api():
    #pip install SpeechRecognition
    import speech_recognition as sr
    AUDIO_FILE = '/data/VoxCeleb1/vox1_train_wav/id11251/AvV4LWBq00g/00001.wav'
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source: 
        #r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.record(source, offset=1)
    r.recognize_google(audio,language='ja-jp',show_all = True)
    IPython.display.Audio(AUDIO_FILE)


# decode opus
def opus_decoder():
    # offline install opuslib(git)
    import opuslib
    import IPython
    channels = 1
    rate = 16000
    frame_size = 960
    decoder = opuslib.Decoder(rate, channels)
    pcms=[decoder.decode(i,frame_size) for i in x]
    return b''.join(pcms)
    # IPython.display.Audio(np.fromstring(b''.join(pcms),np.int16),rate=16000)


