# 海康相机内参标定

采集海康 USB / GigE 相机图像，使用四标签 AprilTag 板标定内参，并输出独立验证结果及离线 HTML 误差报告。

## Part 1：快速开始

### 环境配置

使用 Windows、Conda，并单独安装海康 MVS SDK（含 Python 开发组件）。在项目根目录执行：

```powershell
conda env create -f environment.yml
conda activate opestimation
```

SDK 无法自动定位时，将 `MVS_SDK_PATH` 设置为 MVS 的 Development 目录或包含 Python 绑定的 MvImport 目录。
环境固定使用 `pyapriltags==3.4.3.1`；已有环境仅补装此依赖时执行：

```powershell
python -m pip install --only-binary=:all: --no-deps pyapriltags==3.4.3.1
```

### 项目运行

```powershell
python -m intrinsic devices
python -m intrinsic capture --serial DB2632441
```

仅连接一台相机时可省略序列号；运行前释放 MVS 或其他程序对相机的占用。通过 USB 转网口适配器连接的相机仍属于 GigE。

| 按键 | 操作 |
|---|---|
| Space | 保存标定图，建议 30 张，至少 15 张 |
| V | 保存独立验证图，建议 5～10 张 |
| Q / Esc | 退出采集 |

固定镜头对焦和光圈，覆盖画面中央、边缘及不同倾斜角度，每次停稳后保存。每张图需要完整识别四个标签。
采集启动后会打印会话目录；将下方路径替换为实际会话目录后运行：

```powershell
python -m intrinsic calibrate --session "data/DB2632441/20261102T082210Z_121c2e93"
```

| 标定参数 | 行为 |
|---|---|
| `--validation auto` | 默认；有验证图就验证，没有则明确显示“未验证” |
| `--validation required` | 必须验证；无验证图或验证失败时返回错误 |
| `--validation off` | 关闭独立验证 |
| `--exclude FRAME_ID ...` | 排除指定标定图，不删除原图 |

当前 `pipeline.json` 默认选择 pyapriltags；保留 OpenCV 后端。配置优先级为：显式 CLI 参数 > pipeline 检测配置 > 旧配置默认 OpenCV。未传 CLI 参数时不覆盖配置。

```powershell
python -m intrinsic capture --serial DB2632441 --detector opencv --corner-refinement CONTOUR --corner-window-size 5
python -m intrinsic import-images --images "D:/images" --camera-id DB2632441 --detector pyapriltags
python -m intrinsic compare-detectors --session "data/DB2632441/20261102T111838Z_15994aa3" --schemes configs/compare.json
```

检测配置可在 `pipeline.json` 的 `detector` 节中指定；未写出的参数采用以下默认值：

```json
"detector": {
  "backend": "pyapriltags",
  "opencv": {"corner_refinement": "SUBPIX", "window_size": 5},
  "pyapriltags": {"quad_decimate": 1.0, "refine_edges": 1}
}
```

OpenCV 方法支持 `NONE`、`SUBPIX`、`CONTOUR`、`APRILTAG`，均属于同一后端。pyapriltags 的参数可用 `--quad-decimate`、`--refine-edges 0/1` 覆盖；CLI 检测参数必须属于选中的后端。标签家族统一来自标定板配置。

多算法方案由 [configs/compare.json](configs/compare.json) 单独维护，每项包含 `name`、`backend`、`parameters`；未传 `--schemes` 时读取该文件。源会话只通过 CLI 指定。默认使用各方案共同成功的帧；单会话可用 `--frame-ids` 指定完整固定集合，固定集合中检测失败的方案不会缩小集合标定。

必要单元测试使用合成数据和模拟求解器；真实数据比较必须显式指定会话，不随普通单元测试执行：

```powershell
python -B tests/test_detector_comparison.py
python -B tests/test_detector_comparison.py --session "data/DB2632441/20261102T111838Z_15994aa3" --schemes configs/compare.json
```

## Part 2：输入与输出

### Input

- 相机输入：通过 MVS 接入的海康 USB / GigE 相机。默认采集全幅最大分辨率，退出时恢复临时设置；网口相机自动探测传输包大小，失败时尝试 1500 字节。
- 标定板：`tag36h11`，左上、右上、左下、右下 ID 为 1、2、3、4；黑框外边长 20 mm、净间距 10 mm，整板 70 mm。朝向见 [参考图](docs/calib_board_examples/apriltag_20mm.png)，打印后需实测尺寸。
- 配置：默认读取 [configs/pipeline.json](configs/pipeline.json)，标定板配置在 [configs/boards/apriltag_20mm.json](configs/boards/apriltag_20mm.json)，相机配置按序列号读取 `configs/cameras/<序列号>.json`。命令行参数优先于相机配置；最终配置保存到会话。
- 离线输入：支持 8 位 PNG、JPEG、BMP、TIFF 图像，必须来自同一相机、镜头和成像设置。标定图与验证图通过用途字段区分。

```powershell
# 指定配置、固定曝光和增益
python -m intrinsic capture --serial DB2632441 --config configs/pipeline.json --exposure-us 5000 --gain-db 0
# 沿用现有 ROI；连续自动增益与固定增益参数互斥
python -m intrinsic capture --serial DB2632441 --resolution current --gain-auto continuous
# 导入同一成像设置下的离线图像
python -m intrinsic import-images --images "D:/images" --camera-id DB2632441
```

### Output

数据位于 `data/<序列号>/<采集会话>/`。采集会话名使用 UTC 时间及随机标识；图像与逐帧 JSON 使用相同帧 ID 对应。

| 路径或文件 | 内容 |
|---|---|
| `session.json` | 相机身份、成像设置、分辨率及一份运行配置；实际检测参数集中保存 |
| `images/<帧ID>.png` | 无损原始几何图像；彩色数据在 OpenCV 中按 BGR 通道处理 |
| `frames/<帧ID>.json` | 角点、图像质量指标、标定/验证用途和帧元数据 |
| `results/calib_YYYYMMDDHHmm/` | 每次标定的独立结果目录，使用北京时间；重名时追加序号 |
| `report.html` 与 `previews/` | 离线误差报告及叠加预览；分享时一起复制 |
| `calibration.json` | 标定参数、诊断、逐图位姿和独立验证结果；通过源会话读取运行配置 |
| `intrinsics.json` | 日常使用的相机身份、标定时间、图像尺寸、K、D 和畸变参数顺序 |
| `per_view.csv` | 每张标定/验证图像的误差统计及复查原因 |
| `residuals.csv` | 每个角点的观测坐标和重投影残差 |

HTML 报告由顶部概览和四部分组成：标定质量分析、异常图像复查、逐图结果、标定参数。
角点覆盖与残差 SVG 并排显示，图像按 RMSE 降序排列；逐图表格和详细参数默认折叠。没有独立验证时明确显示“未验证”。

新数据不生成或校验内容哈希；旧会话中的哈希字段可兼容读取但会被忽略。Frame ID、Session ID 和标定/验证用途保持原规则；仍检查格式、尺寸及可读性。不再按像素去重，导入时需自行避免同图同时用于标定和验证。新帧用 `detector_index` 引用会话 `detectors` 中的实际检测参数，不逐帧复制配置。

K 中焦距和主点的单位为 px，D 为无量纲，顺序是 `[k1,k2,p1,p2,k3]`。
角点位于原始图像像素坐标系，原点在左上角，u 向右、v 向下；残差为观测点减投影点。
MRE、RMSE、最大误差和误差标准差的单位为 px，方差为 px²；误差标准差与方差采用样本统计 `ddof=1`。
逐图位姿将标定板坐标变换到相机坐标，平移单位为 m，旋转向量为 Rodrigues 表示。

复查标记沿用配置中的 RMS 阈值，默认 1 px，不自动删除图像。“未触发告警”不代表人工检查通过，像素重投影误差不等于毫米测量精度。
更改镜头、对焦、分辨率或 ROI 后需建立新会话并重新标定。

固定已有内参补做独立验证，或比较两份结果：

```powershell
python -m intrinsic validate --session "会话目录" --result "已有结果/calibration.json"
python -m intrinsic compare --baseline "原结果/calibration.json" --candidate "新结果/calibration.json" --output "新的对比目录"
```

单独验证生成新的 `validate_时间/` 目录；对比生成 `comparison.html` 和 `comparison.json`。
验证失败时仍保留内参并记录原因；对比不会重新标定，验证图不一致时不直接比较验证误差。

`compare-detectors` 默认新建 `data/test_results/detector_compare_<UTC时间及随机标识>/`，也可用 `--output` 指定尚不存在的目录。单会话输出如下；方案目录按配置动态生成，未完成标定的方案只有检测记录：

```text
detector_compare_<时间>/
├── input_manifest.json       # 源会话相对路径、帧 ID/用途、板模型与实验条件
├── comparison.json          # 各方案状态、角点差异及对照图索引
├── comparison.html          # 离线总报告
├── per_view_comparison.csv
├── corner_differences.csv
├── corner_previews/         # 保留角点对照拼图
└── methods/
    └── <方案名>/
        ├── detections.json  # 实际检测参数、版本、逐帧角点和失败原因
        └── calibration.json # K、D、标准差、误差、位姿及诊断
```

Compare 仅在内存中构造只读会话，原图从源会话读取。不输出方法级 `session/`、原图副本、独立报告、全尺寸 overlay、重复内参文件/CSV、实验脚本或审计文件。配置和标定参数分别保存在输入清单及方法记录中，总报告合并展示，不重复写入汇总 JSON。

报告展示算法与实际参数、坐标转换、源会话与帧集合、检测成功率、训练/独立验证误差、K/D/标准差和角点差异。分享整个 Compare 目录即可查看报告及对照图；读取原图仍需保留源会话相对位置。方法级标定记录通过帧 ID 关联检测角点，不保存失效的本地图像路径。
检测成功率统计全部输入帧；主要误差比较使用所有方案共同成功的相同标定/验证帧，不按误差自动剔除。角点差异单位为 px，不是真值误差；参数估计标准差也不等于重复实验稳定性。
共同标定帧不足配置下限或求解失败时，保留检测记录和失败原因，命令返回非零状态，不调整阈值继续拟合。成功率仅针对会话中已保存的图像，不包含采集时未保存的帧。

已有五方案结果 `data/test_results/detector_compare_20261102T142509Z_f8b0c8ea/` 已整理为上述结构，未重新检测或标定。历史 APRILTAG 记录另保留转换前的角点与质量指标；没有独立验证数据。

## Part 3：算法流程

1. 使用 MVS 取流并转换为 Mono8 / BGR8，保留原始像素几何；按配置调用 OpenCV 或 pyapriltags，固定坐标适配后执行共同质量检查，输出兼容的 Observation。
2. 将图像、角点及相机设置保存为采集会话，区分标定图和独立验证图。
3. 使用标定板三维坐标与图像角点，调用 OpenCV 求解针孔模型和 Brown 五参数畸变，计算重投影残差及已有诊断。
4. 独立验证固定 K、D，仅估计每张验证图的标定板位姿，不将验证图用于内参求解。
5. 导出 JSON、CSV 和预览图；展示模块根据已有统计生成 HTML，页面数值格式化不影响保存精度。

pyapriltags 默认使用原始灰度图、`tag36h11`、`quad_decimate=1.0`、`refine_edges=1`，不预先去畸变或估计 Tag 位姿。角点固定按 `[1,0,3,2]` 重排，x、y 各减 0.5 px；OpenCV APRILTAG 沿用原编号，同样各减 0.5 px；NONE/SUBPIX/CONTOUR 不额外平移。转换只在 Detector 中执行一次，先于共同质量检查。
当前 pyapriltags 版本实测 `refine_edges` 在默认设置下会影响角点；该版本绑定将默认 `decode_sharpening=0.25` 截断为实际 `0`，保留此行为并记录实际参数及版本。
本项目使用 tag36h11 标定板。在 DB2632441 相同 17 张图的已有对比中，pyapriltags 的训练 RMSE 为 0.391959 px，低于 OpenCV SUBPIX 的 2.175249 px 和 CONTOUR 的 0.986527 px；当前配置据此选择 pyapriltags。此结果没有独立验证，不能仅因板子使用 AprilTag 就推断所有数据下均更优。

## Part 4：工程结构

```text
Intrinsic/
├── environment.yml       # 可复现的 Conda 环境
├── configs/              # 采集流程、标定板及按序列号区分的相机配置
├── intrinsic/            # 命令行、设备取流、检测与标定
│   ├── cli.py            # 命令入口、导入及检测器对照实验调度
│   ├── board.py          # 标定板几何、两种检测前端与统一质量检查
│   ├── session.py        # 原始会话及不落盘的只读实验视图
│   ├── solver.py         # 内参求解、独立验证和统计计算
│   ├── report.py         # 数据与预览导出、报告生成调度
│   └── report_html.py    # HTML/CSS 展示，不执行标定
├── tests/                # 自动化测试；test_detector_comparison.py 复用正式 Compare 流程
├── docs/                 # 标定板参考图
└── data/                 # 按相机和会话保存的采集数据及结果
```
