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
