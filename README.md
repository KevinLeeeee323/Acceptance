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
- 从固定合格清单中读取 50 条同编号配对数据，按关节名把固体右臂 `action` 的 2050 帧和液体左臂 `action` 的 1100 帧拼成 3150 帧。SVG 虚线在第 2050 帧。三路视频按 `cam_left_wrist`、`cam_high`、`cam_right_wrist` 分别拼接，输出保留原 AV1 帧，每路 105 秒。
- 首次打开某条轨迹、首次播放某路视频时生成对应缓存。默认写入项目下的 `.acceptance-cache/`，可用 `--cache-dir` 改位置。视频可在支持 AV1 的浏览器中播放和拖动进度。三路视频在轨迹图上方显示，加载后显示首帧并保持暂停，通过两个模块之间的统一播放按钮和进度条同步控制，右侧显示当前时长／总时长、当前帧索引／总帧数。帧索引从 0 开始。悬停轨迹只查看数值，点击轨迹或拖动统一进度条，会让三路视频定位到该帧并暂停。
- 图表显示专家参考 action、帧索引和时间戳。前六个关节以 rad 表示，夹爪沿用参考图的 m 标注。模型曲线由推理服务返回，原子动作区间由人工模板提供。

## 固定清单和退出

UI 自动加载 `OverAllInpsect/selected_samples.json` 中的全部 50 条数据，不再抽取 N 条，也不再提供随机种子设置。先以固定抽取种子 42 从合格项中随机选取 50 条，再以固定排序种子 43 排序一次，并写入清单；每次打开保持该顺序。实际验收中仍会重新调用模型推理，不直接使用筛选结果作为新的推理结果。

个人结束验收时，直接关掉浏览器标签页和自己电脑上的 SSH 转发即可。页面上的「退出」按钮会在确认后关闭服务器上的共享服务，所有验收人员的页面都会断开。

单独提取固体某条数据的右臂轨迹：

```bash
/home/hdy/kai0/.venv/bin/python trajectories.py 0 > episode_000000_right_arm.json
```

## 模型推理

使用 `~/kai0` 中的 OpenPI 客户端协议，默认连接 `10.132.90.113:8002` 的 Solid 和 `10.132.90.113:8001` 的 Liquid 服务。启动环境需能导入 `openpi_client`，requirements 指向本机已有源码。需要现在对应服务器上启动推理模型:

```bash
cd kai0 && CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port 8002 policy:checkpoint \
--policy.config=solid_agilex \
--policy.dir=/home/hdy/checkpoints/solid_awbc_99999
```

```bash
# 终端2
cd kai0 && CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py --port 8001 policy:checkpoint \
--policy.config=liquid_agilex \
--policy.dir=/home/hdy/checkpoints/liquid_awbc_119999/liquid_awbc_119999/
```

开始推理后按 Samples 顺序，逐条发送原始观测状态、三路 RGB 图像和逐帧 task_index 对应文本。每次采用模型返回的完整动作段，再从下一段起点的参考观测继续推理，属于基于数据集观测的动作拟合评测。Solid 取右臂，Liquid 取左臂，动作段在任务边界截断。返回值严格检查为有限的 [horizon, 14] actions。

点击停止后等待当前已发出的请求完成，再停止后续请求，停止期间批次按钮保持禁用。开始过推理但尚未完成全部条目时，退出会显示放弃未完成验收数据的确认。全部推理完成后，退出仍要求导出通过核验的最新结果。推理结果在运行时保留于内存，完成的条目同时写入 logs 下的运行目录。刷新可继续查看。每条完成后红线默认用 5 秒展示，可跳过或重播，系统减少动态效果设置会直接展示完整曲线。

七项 RMSE 使用全部 3150 个对应帧计算。总体 RMSE 为 J01–J06 的 RMSE 均方根，单位 rad。J07 单独以 m 展示。

## 人工评定与备注

左侧成功率格子与 Samples、上一条和下一条联动。选择成功或失败后点击「确定」保存并锁定，需要修正时点击「重新评定这一条」。重新评定期间该条恢复为待确认，成功率使用已确认成功的条数除以本批次总条数。全部确认后显示完成提示。

备注在弹出的编辑页输入时自动保存到服务端。评定和备注按固定有序名单组织。仅浏览时保留于服务内存，开始推理后写入运行目录，导出时一并打包。旧版当前浏览器的 localStorage 记录会在首次连接空白服务时迁移。当前格子切换到固体右臂与液体左臂拼接的七维轨迹。

## 原子动作区间

根目录的 `AtomActionsInfo.json` 为第 40 条固液配对数据的人工标注模板，清单内 50 条数据共用其中的 10 个 Solid 和 6 个 Liquid 动作区间。区间按 [起始帧, 终止帧) 解释，Liquid 的局部帧号加 2050 转为拼接帧号，未标注的间隙保留为空白。

服务启动时将模板转换为统一帧号，缓存到 `.acceptance-cache/atomic-actions/`，也支持 `--cache-dir`。点击「生成验收批次」后，七个 SVG 子图自动显示对应底色，顶部显示编号条，底部图例列出动作编号和名称。悬停图例可查看描述，轨迹定位读数显示对应动作名。修改模板后重新启动服务，内容摘要变化会生成新的标注缓存。

## 导出与异常记录

每次服务启动在右上角显示运行 ID，点击「开始推理」后才创建 `logs/<运行ID>/`。批次内所有样本完成推理后，「导出」生成带上海时区时间戳的结果目录和 ZIP。弹窗显示服务器上的绝对路径，可拷贝整个目录或 ZIP。

每个结果包包含 `samples.csv`、`batch_summary.json`、每条样本的七维轨迹 PNG、评定与成功率、备注、异常记录，以及一份推理配置、轨迹对齐说明和有效动作判据表。CSV 删除了 solid_dataset、solid_episode、liquid_dataset、liquid_episode 四列。推理配置由 ssh 3090 核实后保存在 `inference_configuration.json`，模型部署变更后应更新该文件。动作判据表供人工核验使用。

异常记录可在每条数据上分别编辑，包含现象选项、其他现象说明、原因与具体情况、分钟精度发生时间、处理人、跟进解决方式，服务端自动保存，导出时合并为 `异常记录.json`。发生时间按北京时间（Asia/Shanghai）填写，“当下”按钮也按该时区生成分钟级时间。

导出校验包括 PNG 解码验证、各文件 SHA-256 和 ZIP 完整性。未开始推理时可以正常退出。开始后若未完成全部推理，可在确认放弃保存后退出；若已全部完成，须将最新内容导出并通过核验后退出。导出后修改评定、备注或异常记录，需重新导出。新批次开始前会核验前一次运行的结果。

已完成的运行快照可使用 `--restore-run <快照JSON路径>` 恢复。启动时逐条校验七维轨迹形状和 RMSE 与当前参考数据的一致性，恢复后需重新导出才能退出。本次升级保存的旧版 8092 结果快照位于 `.acceptance-cache/migrations/8092_completed_results.json`。

导出包中的 `导出文件说明.md` 解释各文件用途、`results/episode_XXXXXX.json` 的 [7, 3150] 数据结构及样本／图像对应关系。导出包包含本批完整的数值推理结果。运行信息由汇总和评定等文件覆盖，不另生成“运行记录.json”。自动保存的内部状态写入 `.acceptance-cache/run-state/<运行ID>/评定缓存.json`。


## 扩大范围筛选

`inspect_all.py` 扫描两套数据集共同存在的编号，先检查 0–69，再按 70–79、80–89、90–95 扩展。每条新检查的完整模型预测、RMSE、耗时和异常写入 `OverAllInpsect/`。脚本可根据检查记录断点续跑，已完成条目沿用该次结果。严格以六关节总体 RMSE <0.1 rad 为合格条件，先从按编号升序排列的合格数据中用种子 42 随机抽取 50 条，再将入选编号按升序排列并用种子 43 随机打乱，生成展示清单；两个种子分别记录在清单的 selection_seed 和 order_seed 中。若全部编号检查完仍不足 50 条，记录 exhausted 状态。

筛选期间，配对轨迹和三路拼接视频会预先写入 `.acceptance-cache/`。可使用 `/home/hdy/kai0/.venv/bin/python inspect_all.py` 运行检查。阈值筛选的当次结果与之后验收重新推理的结果分别保存。
