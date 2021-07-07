# audio

### 工具

* librosa
* pyaudio
* scipy.io.wave
* pydub.AudioSegment
* PyAstronomy
* pyAudioAnalysis

### 

### Music information retrieval \(MIR\)

### 声纹

Speaker verification \(SV\) is the process of verifying whether an utterance belongs to a specific speaker, based on that speaker’s known utterances \(i.e., enrollment utterances\), with applications such as Voice Match.

浊音的前三个共振峰（formant）对说话人个性特征体现较明显，而前两个对于区别语音内容体现明显。

Depending on the restrictions of the utterances used for enrollment and verification, speaker verification models usually fall into one of two categories: text-dependent speaker verification \(TD-SV\) and text-independent speaker verification \(TI-SV\).

speaker recognition

speaker diarization

说话人识别\(SRE\)

声纹识别\(VPR\)

[i-vector](http://cslt.riit.tsinghua.edu.cn/mediawiki/images/c/cb/131104-ivector-microsoft-wj.pdf) \[N. Dehak, 2011\]; GMM-UBM framework \[D. A. Reynolds, 2000\] GMM-UBM\(Universal Background Model, 通用背景模型\) -&gt;Joint Factor Analysis, JFA算法 -&gt; 全变量系统\(Total Variability\) -&gt; i-vector  
𝑀 = 𝑚 + 𝑇w, T is a low rank 𝐶𝐹 × 𝑅 subspace that contains the eigenvectors with the largest eigenvalues of **total variability \(tv\) covariance matrix**. w~𝑁\(0,𝐼\)

DNN : x-vector, \(via 人脸识别\) Triplet loss, center loss etc.  
Margin Loss such as ASoftmax/ArcSoftmax\(均匀超球面类间距\)

声纹反作弊

ref:  
音色与频率: 谐波

### 语音理解、生成

Connectionist Temporal Classification \(CTC\): TDNN \(OCR task as example\)

TTS: WaveNet \(NN vocoder\) to TocoTron2 \(32X32X2 to 128X128X2\) to WaveRNN \(1-D conv to RNN, only relying on the last sample\) to LPCNet \(a WaveRNN variant, efficient speech synthesis\)、LAS \(Listen, Attend and Spell\) network

语音转化/转写 - Voice Conversion: StarGAN

### 音频质量 audio quality

PESQ - FR\(Full reference\)

looking for a NR\(Non reference\) implementation

Mean Opinion Score \(MOS\)

信噪比估计 PyAstronomy, WADA SNR est

Noise-Estimation Algorithms

1. 递归平均噪声算法
2. 最小值跟踪算法
3. 直方图噪声估计算法

### 音频特征 [python](https://www.kaggle.com/varanr/audio-feature-extraction)

**功率谱密度与频率关系** （粉红噪音-低频高功率，清音功率远小于浊音）

**Formant \(共振峰\)** - \(基频 - 谐振，泛音主要频率峰\)

**Pitch（基频）**- 声带振动 - 准周期浊音（vs清音，声带不振动，元音都是浊音，辅音都有）

**光谱质心**

**光谱衰减**

\*\*\*\*[**Filter bank\(fbank\)和mfcc by python**](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)

**色度频率\(八度\)**

x, sr = librosa.load\('../simple\_piano.wav'\)

hop\_length = 512

chromagram = librosa.feature.chroma\_stft\(x, sr=sr, hop\_length=hop\_length\)

librosa.display.specshow\(chromagram, x\_axis='time', y\_axis='chroma', hop\_length=hop\_length, cmap='coolwarm'\)

### 人声伴奏分离提取

```python
S_full, phase = librosa.magphase(librosa.stft(x[10000:80000]))
# nearest neighbor
S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))
S_filter = np.minimum(S_full, S_filter)

# instrumentation mask margin and vocals mask margin
margin_i, margin_v = 2, 10
power = 2
# softmask(X, X_ref, power=1, split_zeros=False)
#   `M = X**power / (X**power + X_ref**power)`
mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)
S_foreground = mask_v * S_full
S_background = mask_i * S_full

# listern to the seperation result
y=librosa.istft(S_background*phase)

```





