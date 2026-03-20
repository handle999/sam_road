已经让GPT帮我把bash改写为cmd，所以可以在windows上执行

```cmd
eval_schedule.cmd
```

注意需要，路径书写
另外，需要下载额外的依赖
```shell
pip install svgwrite
```

另外，topo用到了`HopcroftKarp`，但是这个库始终无法安装（用到`setuptools`版本很低，而且`github`删库）
```shell
# 这个是topo的，暂时先跳过
pip install hopcroftkarp
```
所以考虑用networkx的替代
在`spacenet_metrics/topo/`实现了`HopcroKarp.py`，并且修改`topo.py`中的实现


powershell -NoProfile -Command "& { .\eval_schedule.cmd 2>&1 | Tee-Object -FilePath sam_official_ckpt.log }"
powershell -NoProfile -Command "& { .\eval_sam.cmd 2>&1 | Tee-Object -FilePath sam_road_my_ep10_copy.log }"

powershell -NoProfile -Command "& { .\eval_schedule.cmd 2>&1 | Tee-Object -FilePath sam_official_my_ckpt_ep20.log }"
powershell -NoProfile -Command "& { .\eval_sam.cmd 2>&1 | Tee-Object -FilePath sam_road_contra_lambda001_ep20.log }"


# 2026/03/21

修改了一些代码，因为原来的测试太慢了，所以用gemini 3 pro写了一些并行的东西，同时整理结构，加了tqdm展示

（这就不得不感慨一句了，上次改这个代码是2025.6，差不多9个月之前，当时还是GPT一骑绝尘，现在大模型已经百花齐放了，未来什么样子呢）

现在的结构仍然是原来的分层，这样更有利于win-linux跨系统迁移
- win: eval_schedule.cmd 
    - -> apls.cmd -> apls.py -> go
    - -> topo.cmd -> topo_parallel.py -> topo/eval_parallel.py   (原来是topo.py -> topo/main.py)
    - 注意：我注释掉了`spacenet_metrics\topo\topo.py`中的`Line 235-236`，有两个print干扰tqdm，具体信息分别是：
        - print(len(result))：统计了生成的起始点数量，生成的起始点数量会影响后续评测的结果，这里统计了生成的起始点数量
        - print("Skipped tunnels ", tunnel_skip_num)：隧道内的道路在卫星图上是不可见的，所以作者不将隧道内的点作为评测的种子点，这里统计了被跳过的隧道点数量
- linux: eval_schedule.bash
    - -> apls.cmd -> apls.py -> go
    - -> topo.bash -> topo_parallel.py -> topo/eval_parallel.py

具体魔改的部分比较多，基本上涉及到的全部变动了，实现了压榨CPU核心级别（仅保留MAX-2）的并行，速度确实是从原来的20min提高到了0.5min

细节就不一一细说了，总而言之结果是对的（和原来的metric吭哧吭哧老半天计算出来一致）

