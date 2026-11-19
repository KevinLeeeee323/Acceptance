# 具身化学项目轨迹验收

## 1. 在服务器上启动服务

当前服务器的 `/home/hdy/kai0/.venv` 已安装 PyArrow 和 PyAV，可以直接用它启动：

```bash
cd /home/hdy/Acceptance
/home/hdy/kai0/.venv/bin/python app.py
```

服务默认监听服务器的 `127.0.0.1:8092`。使用上述命令启动时，保持运行服务的终端打开。

项目里的 `.venv` 需要手动创建。只有在需要给项目单独配置 Python 环境时，才在服务器的 `/home/hdy/Acceptance` 下执行下面三行。项目依赖 Python 3.10+。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

## 2. 在各自电脑上连接

每位验收人员在自己电脑的终端运行下面的命令，将 `user@server` 换成自己的 SSH 用户名和服务器地址：

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8092:127.0.0.1:8092 user@server
```

保持这个终端打开，再用自己电脑上的浏览器打开 [验收页面](http://127.0.0.1:8092/)。多位验收人员可各自在自己的电脑上建立转发，访问同一个服务。访问端只需要 SSH 客户端和浏览器。

## 拼接和缓存

- 固体默认读取 `/data2/organ_data_le/solid/measure_solid_0630`。液体优先读取 `/data/organ_data_le/liquid/measure_liquid_full_0622`，若该路径未挂载则尝试 `/data2` 下的同名路径。可用 `--dataset`、`--liquid-dataset` 分别指定。
- 在编号 0–49 内配对，按关节名把固体右臂 `action` 的 2050 帧和液体左臂 `action` 的 1100 帧拼成 3150 帧。SVG 虚线在第 2050 帧。三路视频按 `cam_left_wrist`、`cam_high`、`cam_right_wrist` 分别拼接，输出保留原 AV1 帧，每路 105 秒。
- 首次打开某条轨迹、首次播放某路视频时生成对应缓存。默认写入项目下的 `.acceptance-cache/`，可用 `--cache-dir` 改位置。视频可在支持 AV1 的浏览器中播放和拖动进度。三路视频在轨迹图上方显示，加载后显示首帧并保持暂停，通过两个模块之间的统一播放按钮和进度条同步控制，右侧显示当前时长／总时长、当前帧索引／总帧数。帧索引从 0 开始。悬停轨迹只查看数值，点击轨迹或拖动统一进度条，会让三路视频定位到该帧并暂停。
- 图表显示专家参考 action、帧索引和时间戳。前六个关节以 rad 表示，夹爪沿用参考图的 m 标注。模型曲线与原子动作区间待接入。

## 抽样和退出

抽样条数默认 20。首次抽样使用 Seed 42。「生成验收批次」显示样本和轨迹。「换一批数据」将种子加一并预览新名单，接着点击「生成验收批次」显示该名单对应的轨迹。若选择全部样本，换批会改变顺序。刷新保留当前批次及选中条目。

个人结束验收时，直接关掉浏览器标签页和自己电脑上的 SSH 转发即可。页面上的「退出」按钮会在确认后关闭服务器上的共享服务，所有验收人员的页面都会断开。

单独提取固体某条数据的右臂轨迹：

```bash
/home/hdy/kai0/.venv/bin/python trajectories.py 0 > episode_000000_right_arm.json
```
