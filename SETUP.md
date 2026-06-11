安装颇费周折

用了之前autodl的环境，下载了conda

但是有一个问题，如果用yml，服务器的官方pip会卡死，所以应当使用清华源

因此使用conda和pip分开的安装，一个是`conda_only.yml`，一个是`pip_requirements.txt`，分别来自于

```shell
conda env export --from-history > conda_only.yml
pip freeze > pip_requirements.txt
```

值得注意的是，`pip freeze`的有奇怪的本地包，所以用了`conda list | grep pypi`输出环境，AI加持整理格式，生成符合条件的req

然后有下面几种特殊情况

### 1. 包冲突

有些环境冲突，比如

```shell
ERROR: Cannot install -r pip_requirements.txt (line 131), -r pip_requirements.txt (line 83), -r pip_requirements.txt (line 84) and protobuf==6.31.1 because these package versions have conflicting dependencies.

The conflict is caused by:
    The user requested protobuf==6.31.1
    onnx 1.18.0 depends on protobuf>=4.25.1
    onnxruntime 1.22.1 depends on protobuf
    tensorboard 2.15.1 depends on protobuf<4.24 and >=3.19.6

Additionally, some packages in these conflicts have no matching distributions available for your environment:
    protobuf

To fix this you could try to:
1. loosen the range of package versions you've specified
2. remove package versions to allow pip to attempt to solve the dependency conflict

ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts
```

所以把其中一些的版本直接放掉了

```shell
# line 85
opencv-python
# line 131
tensorboard
tensorboard-data-server
```

### 2. 无法安装

有些东西不是清华源能够安装的，要从官网（比如pypi，或者pyg），但是完全从官网又太慢卡死，这些选择单独安装

```shell
# line 138
torch==2.1.2+cu121
torch-cluster==1.6.3+pt21cu121
torch-geometric==2.4.0
torch-scatter==2.1.2+pt21cu121
torch-sparse==0.6.18+pt21cu121
torch-spline-conv==1.2.2+pt21cu121
torchmetrics==1.7.3
torchvision==0.16.2+cu121
```

```shell
# 例如 CUDA 12.1
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# 卸载旧的
pip uninstall -y torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

# 安装对应 CUDA wheel
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
pip install torch-geometric==2.4.0
```

最后还原req文件，再执行一遍就行

```shell
pip install -r pip_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
