# What's this code?
    このコードは、入力された画像から(x,y, x,y, x,y, x,y)の四点を推論することができるCNNです。
    ！！！！------グレースケール（1ch入力）専用です。------！！！！

    学習、推論、onnx出力などの実行機能はすべてmain.pyにあります。「python main.py」でそのまま実行してください。
    細かな機能関数はutils.pyに記述してあります。utils.pyを実行することはありません。
    設定類はすべてconfg.pyに記述してあります。config.pyを実行することはありません。
    ネットワークの定義はnetsフォルダ内に定義してあります。main.pyの先頭で使用するネットワークを選択できます。
    
    環境構築は以下に示してあります。

    Rabelmeを用いてアノテーションしてください。四つのポイントの名前はデフォルトで[1, 2, 3, 4]です。config.pyで変えられます。変更の際はアノテーションの名前も変えてください。

    データセットは、画像をdatasetフォルダに、jsonファイルをjsonフォルダへ格納してください。jsonフォルダがなくても画像があれば学習できます。jsonファイル内の"ImagePath"だけは画像と対応しているのか確認してください。ここを参照してdatasetフォルダ内の画像とjsonファイルを紐づけています。

    アノテーションがちゃんと反映されているかは、main.pyを実行して「log」フォルダ内の、「input_img」フォルダを確認してください。入力画像にアノテーションが反映された画像が保存されます。データ拡張に関しても、「input_img」を確認することで正常に拡張されているのかが確認できます。

    コード使用時にはvscodeの折り畳み機能を使って関数を折りたたんで、都度開く感じをお勧めします。

    一応関数の上部にinput、outputを記載したり、main.pyの一番下に流れを書いてますが、いろいろ変更したり、そもそもAIが作成したので、あっている保証はありません。もう一度AIに作成をお願いしたほうがいいかも。

# What's to_160.py?
    これは指定したフォルダ内の画像をすべて中心から上下左右に80ピクセル切り取って160x160にするコードです。コード下部で対象フォルダと出力フォルダを指定できます。







# 修正待ち
    データセットが少なすぎて、ヒートマップの関数が正常に動作しているのか分からない。

# 追加予定機能
    量子化機能の追加。
    gap8用Cコードの生成機能の追加。

# 修正・追加中事項
    正規データセットの追加
  

# 注意事項

    2025-06-10 14:20:32,471 INFO Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
    2025-06-10 14:20:32,472 INFO Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.

    学習時などにこのような警告文が出るが、無視して問題ない。ポイントの名前が数字だと、グラフ表示時に注意が入るらしい。

# anaconda を用いた環境構築

```
#------------------------------------
#　環境構築(anaconda3)※stable version
#------------------------------------
conda create -n gate python=3.9.21
conda activate gate

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install matplotlib=3.9.2
conda install onnx=1.17.0
conda install pillow=11.0.0
conda install scikit-learn=1.6.1
conda install seaborn=0.13.2
conda install tensorboard=2.19.0
conda install tqdm=4.67.1

pip install thop==0.1.1-2209072238
pip install torchinfo==1.8.0
pip install onnx-graphsurgeon==0.5.8
pip install opencv-python==4.11.0.86

#---------------------------------

#---------------------------------
#　vscode拡張機能他
#---------------------------------
Better Comments　←入れて下さい

Evondev-Indnet Rainbow Palettes ←おすすめ
Error Lens ←おすすめ
indent-randbow　←おすすめ

#---------------------------------




以下stable環境詳細



# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main
_openmp_mutex             5.1                       1_gnu
absl-py                   2.1.0            py39h06a4308_0
blas                      1.0                         mkl
brotli-python             1.0.9            py39h6a678d5_9
bzip2                     1.0.8                h5eee18b_6
c-ares                    1.19.1               h5eee18b_0
ca-certificates           2025.2.25            h06a4308_0
certifi                   2025.1.31        py39h06a4308_0
charset-normalizer        3.4.1                    pypi_0    pypi
coloredlogs               15.0.1                   pypi_0    pypi
contourpy                 1.2.1            py39hdb19cb5_1
cuda-cudart               12.4.127             h99ab3db_0
cuda-cudart_linux-64      12.4.127             hd681fbe_0
cuda-cupti                12.4.127             h6a678d5_1
cuda-libraries            12.4.1               h06a4308_1
cuda-nvrtc                12.4.127             h99ab3db_1
cuda-nvtx                 12.4.127             h6a678d5_1
cuda-opencl               12.4.127             h6a678d5_0
cuda-runtime              12.4.1               hb982923_0
cuda-version              12.4                 hbda6634_3
cycler                    0.11.0             pyhd3eb1b0_0
cyrus-sasl                2.1.28               h52b45da_1
expat                     2.6.4                h6a678d5_0
ffmpeg                    4.3                  hf484d3e_0    pytorch
filelock                  3.13.1           py39h06a4308_0
flatbuffers               25.2.10                  pypi_0    pypi
fontconfig                2.14.1               h55d465d_3
fonttools                 4.55.3           py39h5eee18b_0
freetype                  2.13.3               h4a9f257_0
fsspec                    2024.6.1                 pypi_0    pypi
gmp                       6.3.0                h6a678d5_0
gmpy2                     2.2.1            py39h5eee18b_0
gnutls                    3.6.15               he1e5248_0
grpcio                    1.71.0           py39h6a678d5_0
huggingface-hub           0.30.1                   pypi_0    pypi
humanfriendly             10.0                     pypi_0    pypi
icu                       73.1                 h6a678d5_0
idna                      3.10                     pypi_0    pypi
importlib-metadata        8.5.0            py39h06a4308_0
importlib_resources       6.4.0            py39h06a4308_0
intel-openmp              2023.1.0         hdb19cb5_46306
jinja2                    3.1.4                    pypi_0    pypi
joblib                    1.5.1                    pypi_0    pypi
jpeg                      9e                   h5eee18b_3
kiwisolver                1.4.4            py39h6a678d5_0
krb5                      1.20.1               h143b758_1
lame                      3.100                h7b6447c_0
lcms2                     2.16                 hb9589c4_0
ld_impl_linux-64          2.40                 h12ee557_0
lerc                      4.0.0                h6a678d5_0
libabseil                 20250127.0      cxx17_h6a678d5_0
libcublas                 12.4.5.8             h99ab3db_1
libcufft                  11.2.1.3             h99ab3db_1
libcufile                 1.9.1.3              h99ab3db_1
libcups                   2.4.2                h2d74bed_1
libcurand                 10.3.5.147           h99ab3db_1
libcurl                   8.12.1               hc9e6f67_0
libcusolver               11.6.1.9             h99ab3db_1
libcusparse               12.3.1.170           h99ab3db_1
libdeflate                1.22                 h5eee18b_0
libedit                   3.1.20230828         h5eee18b_0
libev                     4.33                 h7f8727e_1
libffi                    3.4.4                h6a678d5_1
libgcc-ng                 11.2.0               h1234567_1
libgfortran-ng            11.2.0               h00389a5_1
libgfortran5              11.2.0               h1234567_1
libglib                   2.78.4               hdc74915_0
libgomp                   11.2.0               h1234567_1
libgrpc                   1.71.0               h2d74bed_0
libiconv                  1.16                 h5eee18b_3
libidn2                   2.3.4                h5eee18b_0
libjpeg-turbo             2.0.0                h9bf148f_0    pytorch
libnghttp2                1.57.0               h2d74bed_0
libnpp                    12.2.5.30            h99ab3db_1
libnvfatbin               12.4.127             h7934f7d_2
libnvjitlink              12.4.127             h99ab3db_1
libnvjpeg                 12.3.1.117           h6a678d5_1
libpng                    1.6.39               h5eee18b_0
libpq                     17.4                 hdbd6064_0
libprotobuf               5.29.3               hc99497a_0
libre2-11                 2024.07.02           h6a678d5_0
libssh2                   1.11.1               h251f7ec_0
libstdcxx-ng              11.2.0               h1234567_1
libtasn1                  4.19.0               h5eee18b_0
libtiff                   4.5.1                hffd6297_1
libunistring              0.9.10               h27cfd23_0
libuuid                   1.41.5               h5eee18b_0
libwebp-base              1.3.2                h5eee18b_1
libxcb                    1.15                 h7f8727e_0
libxkbcommon              1.0.1                h097e994_2
libxml2                   2.13.5               hfdd30dd_0
llvm-openmp               14.0.6               h9e868ea_0
lz4-c                     1.9.4                h6a678d5_1
markdown                  3.4.1            py39h06a4308_0
markdown-it-py            3.0.0                    pypi_0    pypi
markupsafe                2.1.5                    pypi_0    pypi
matplotlib                3.9.2            py39h06a4308_1
matplotlib-base           3.9.2            py39hbfdbfaf_1
mdurl                     0.1.2                    pypi_0    pypi
mkl                       2023.1.0         h213fc3f_46344
mkl-service               2.4.0            py39h5eee18b_2
mkl_fft                   1.3.11           py39h5eee18b_0
mkl_random                1.2.8            py39h1128e8f_0
mpc                       1.3.1                h5eee18b_0
mpfr                      4.2.1                h5eee18b_0
mpmath                    1.3.0            py39h06a4308_0
mysql                     8.4.0                h721767e_2
ncurses                   6.4                  h6a678d5_0
nettle                    3.7.3                hbbd107a_1
networkx                  3.2.1            py39h06a4308_0
numpy                     1.26.3                   pypi_0    pypi
numpy-base                1.26.4           py39hb5e798b_0
nvidia-cublas-cu12        12.6.4.1                 pypi_0    pypi
nvidia-cuda-cupti-cu12    12.6.80                  pypi_0    pypi
nvidia-cuda-nvrtc-cu12    12.6.77                  pypi_0    pypi
nvidia-cuda-runtime-cu12  12.6.77                  pypi_0    pypi
nvidia-cudnn-cu12         9.5.1.17                 pypi_0    pypi
nvidia-cufft-cu12         11.3.0.4                 pypi_0    pypi
nvidia-curand-cu12        10.3.7.77                pypi_0    pypi
nvidia-cusolver-cu12      11.7.1.2                 pypi_0    pypi
nvidia-cusparse-cu12      12.5.4.2                 pypi_0    pypi
nvidia-cusparselt-cu12    0.6.3                    pypi_0    pypi
nvidia-ml-py              12.570.86                pypi_0    pypi
nvidia-nccl-cu12          2.21.5                   pypi_0    pypi
nvidia-nvjitlink-cu12     12.6.85                  pypi_0    pypi
nvidia-nvtx-cu12          12.6.77                  pypi_0    pypi
nvitop                    1.4.2                    pypi_0    pypi
ocl-icd                   2.3.2                h5eee18b_1
onnx                      1.17.0                   pypi_0    pypi
onnx-graphsurgeon         0.5.8                    pypi_0    pypi
onnxruntime               1.19.2                   pypi_0    pypi
onnxsim                   0.4.36                   pypi_0    pypi
opencv-python             4.11.0.86                pypi_0    pypi
openh264                  2.1.1                h4ff587b_0
openjpeg                  2.5.2                he7f1fd0_0
openldap                  2.6.4                h42fbc30_0
openssl                   3.0.16               h5eee18b_0
packaging                 24.2             py39h06a4308_0
pandas                    2.2.3                    pypi_0    pypi
pcre2                     10.42                hebb0a14_1
pillow                    11.0.0                   pypi_0    pypi
pip                       25.0             py39h06a4308_0
protobuf                  6.30.2                   pypi_0    pypi
psutil                    7.0.0                    pypi_0    pypi
py-cpuinfo                9.0.0                    pypi_0    pypi
pygments                  2.19.1                   pypi_0    pypi
pyparsing                 3.2.0            py39h06a4308_0
pyqt                      6.7.1            py39h6a678d5_0
pyqt6-sip                 13.9.1           py39h5eee18b_0
pysocks                   1.7.1            py39h06a4308_0
python                    3.9.21               he870216_1
python-dateutil           2.9.0post0       py39h06a4308_2
pytorch                   2.4.0           py3.9_cuda12.4_cudnn9.1.0_0    pytorch
pytorch-cuda              12.4                 hc786d27_7    pytorch
pytorch-mutex             1.0                        cuda    pytorch
pytz                      2025.2                   pypi_0    pypi
pyyaml                    6.0.2            py39h5eee18b_0
qtbase                    6.7.3                hdaa5aa8_0
qtdeclarative             6.7.3                h6a678d5_0
qtsvg                     6.7.3                he621ea3_0
qttools                   6.7.3                h80c7b02_0
qtwebchannel              6.7.3                h6a678d5_0
qtwebsockets              6.7.3                h6a678d5_0
re2                       2024.07.02           hdb19cb5_0
readline                  8.2                  h5eee18b_0
regex                     2024.11.6                pypi_0    pypi
requests                  2.32.3           py39h06a4308_1
rich                      14.0.0                   pypi_0    pypi
safetensors               0.5.3                    pypi_0    pypi
scikit-learn              1.6.1                    pypi_0    pypi
scipy                     1.12.0           py39h5f9d8c6_0
seaborn                   0.13.2                   pypi_0    pypi
setuptools                75.8.0           py39h06a4308_0
sip                       6.10.0           py39h6a678d5_0
six                       1.16.0             pyhd3eb1b0_1
sqlite                    3.45.3               h5eee18b_0
sympy                     1.13.1                   pypi_0    pypi
tbb                       2021.8.0             hdb19cb5_0
tensorboard               2.19.0           py39h06a4308_0
tensorboard-data-server   0.7.0            py39h52d8a92_1
thop                      0.1.1-2209072238          pypi_0    pypi
threadpoolctl             3.6.0                    pypi_0    pypi
tk                        8.6.14               h39e8969_0
tokenizers                0.15.2                   pypi_0    pypi
tomli                     2.0.1            py39h06a4308_0
torchaudio                2.6.0+cu126              pypi_0    pypi
torchinfo                 1.8.0                    pypi_0    pypi
torchsummary              1.5.1                    pypi_0    pypi
torchtriton               3.0.0                      py39    pytorch
torchvision               0.21.0+cu126             pypi_0    pypi
tornado                   6.4.2            py39h5eee18b_0
tqdm                      4.67.1           py39h2f386ee_0
transformers              4.38.2                   pypi_0    pypi
triton                    3.2.0                    pypi_0    pypi
typing_extensions         4.12.2           py39h06a4308_0
tzdata                    2025.2                   pypi_0    pypi
ultralytics               8.3.123                  pypi_0    pypi
ultralytics-thop          2.0.14                   pypi_0    pypi
unicodedata2              15.1.0           py39h5eee18b_1
urllib3                   2.3.0            py39h06a4308_0
werkzeug                  3.1.3            py39h06a4308_0
wheel                     0.45.1           py39h06a4308_0
xcb-util-cursor           0.1.4                h5eee18b_0
xz                        5.6.4                h5eee18b_1
yaml                      0.2.5                h7b6447c_0
zipp                      3.21.0           py39h06a4308_0
zlib                      1.2.13               h5eee18b_1
zstd                      1.5.6                hc292b87_0
```
