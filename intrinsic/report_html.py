"""Offline HTML presentation of existing calibration statistics; no fitting or export."""
from __future__ import annotations

from html import escape
import re


STYLE = """
:root{color-scheme:light;font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif;color:#243247;background:#f4f6f8}
*{box-sizing:border-box}body{margin:0;font-size:14px;line-height:1.65}main{max-width:1180px;margin:auto;padding:32px 24px 48px}
h1,h2,h3,p,figure{margin:0}h1{font-size:26px;font-weight:650}h2{font-size:19px;margin-bottom:16px}h3{font-size:15px;margin-bottom:10px}
a{color:#245a94;text-underline-offset:3px}a:hover{color:#163c68}.muted,figcaption{color:#647084;font-size:13px}.intro{margin:8px 0 20px}
header,section{background:#fff;border:1px solid #dce2e8;border-radius:8px;padding:24px;margin-bottom:20px;min-width:0}
.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.metric{padding:16px;background:#f7f9fb;border:1px solid #e3e8ee;border-radius:6px}
.metric dt{color:#59677a;font-size:13px}.metric dd{margin:6px 0 2px;font-size:26px;line-height:1.35;font-weight:650;font-variant-numeric:tabular-nums}
.metric small{font-size:13px;font-weight:400}.metric.attention{border-color:#ddc18d;background:#fffbf2}.metric.status dd{font-size:21px}
dl{margin:0}.note{border-left:3px solid #d5ac62;padding:9px 12px;background:#fffbf2;margin-top:16px;font-size:13px}
.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:10px 12px;border-bottom:1px solid #e4e8ed;text-align:right;vertical-align:top;white-space:nowrap}th{font-weight:600;color:#4a586b;background:#f7f9fb}
th:first-child,td:first-child{text-align:left}tbody tr:last-child td{border-bottom:0}.caption{margin:10px 0 14px;font-size:13px;color:#647084}
.charts,.parameters{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.charts figure{border:1px solid #e3e8ee;border-radius:6px;padding:14px;min-width:0}
svg{display:block;width:100%;height:auto;max-height:310px}figcaption{margin-top:8px}ul{margin:8px 0;padding-left:22px}.diagnostics{margin:16px 0}
details{margin-top:16px}summary{cursor:pointer;font-weight:550;color:#30465f;padding:8px 0}summary:hover{color:#245a94}details[open]>summary{margin-bottom:10px}
.review-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.review-card{border:1px solid #e0e5eb;border-radius:6px;overflow:hidden;min-width:0}
.review-card img{display:block;width:100%;height:158px;object-fit:contain;background:#f3f5f7}.review-body{padding:12px 14px}.frame-id{font-family:ui-monospace,Consolas,monospace;overflow-wrap:anywhere}
.review-values{margin:8px 0 4px;font-size:13px}.reason{font-size:12px;color:#715831}.badge{display:inline-block;border-radius:4px;padding:1px 7px;font-size:12px;background:#f0f3f6;color:#58677b}
.badge.warning{background:#fff1d6;color:#785217}.worst{margin-top:16px;font-size:13px}.scroll-table{max-height:540px;overflow:auto}.scroll-table thead th{position:sticky;top:0}
.matrix{width:auto;margin:12px 0 20px;border-left:2px solid #8b97a8;border-right:2px solid #8b97a8}.matrix td{text-align:right;border:0;padding:6px 16px}
.parameters>div{min-width:0}.parameters td,.parameters th{padding:9px}.parameters-note{margin-top:14px}.source{overflow-wrap:anywhere}footer{color:#6b7583;font-size:12px;padding:0 6px}
@media(max-width:800px){main{padding:16px 12px}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.charts,.parameters{grid-template-columns:1fr}.review-grid{grid-template-columns:repeat(2,minmax(0,1fr))}header,section{padding:18px}}
@media(max-width:480px){.review-grid{grid-template-columns:1fr}.metric{padding:12px}.metric dd{font-size:23px}h1{font-size:23px}}
@media print{body{background:#fff}main{max-width:none;padding:0}header,section{break-inside:avoid;border-radius:0}a{color:inherit}}
"""


def _number(value, digits=4):
    if value is None:
        return "—"
    return f"{value:.6e}" if value != 0 and abs(value) < 10 ** -digits else f"{value:.{digits}f}"


def render_detector_comparison_html(comparison: dict) -> str:
    """Compact offline results; configuration and diagnostics stay in linked JSON."""
    def table(headers, rows):
        head = "".join(f"<th>{escape(str(label))}</th>" for label in headers)
        body = "".join("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>" for row in rows)
        return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
    def link(path, label):
        return f'<a href="{escape(path, quote=True)}">{escape(str(label))}</a>'
    def text(value):
        return escape(str(value))
    algorithms, results_rows, sections = [], [], []
    for ds in comparison["datasets"]:
        names = [scheme["name"] for scheme in ds["schemes"]]
        prefix = "" if ds["directory"] == "." else ds["directory"] + "/"
        camera = ds["camera"]
        for scheme in ds["schemes"]:
            name = scheme["name"]
            detection = ds["detection"][name]
            actual = detection["detector"]
            method = actual.get("corner_refinement", "AprilTag 原生边缘精化")
            algorithms.append([text(camera["serial"]), text(name), text(scheme["backend"]), text(method),
                f'{detection["success_count"]}/{detection["total"]}',
                link(f'{prefix}methods/{name}/detections.json', 'detections.json')])
            result = ds["results"][name]
            train, val = result.get("training_metrics") or {}, result.get("validation_metrics") or {}
            status = "未验证"
            if result["status"] != "completed":
                status = "未完成标定"
            elif result.get("validation_status") == "evaluated":
                status = "已独立验证"
            elif result.get("validation_status") == "failed":
                status = "独立验证失败"
            results_rows.append([text(camera["serial"]), text(name),
                *[_number(train.get(k), 6) for k in ("mre_px", "rmse_px", "max_error_px")],
                *[_number(val.get(k), 6) for k in ("mre_px", "rmse_px")], status])
        differences = []
        for name, references in ds.get("corner_comparisons", {}).items():
            for reference, item in references.items():
                metrics = item.get("metrics") or {}
                differences.append([text(name), text(reference), str(item["compared_frame_count"]),
                    *[_number(metrics.get(k), 6) for k in ("mre_px", "rmse_px", "max_error_px")],
                    "完整" if item.get("status") == "complete" else "不完整"])
        parameter_rows = []
        for i, label in enumerate(("fx (px)", "fy (px)", "cx (px)", "cy (px)", "k1", "k2", "p1", "p2", "k3")):
            row = [label]
            for name in names:
                result = ds["results"][name]
                if result["status"] != "completed":
                    row.append("—")
                    continue
                value = [result["K"][0][0], result["K"][1][1], result["K"][0][2], result["K"][1][2], *result["D"]][i]
                row.append(f'{_number(value, 6)} ± {_number(result["std_intrinsics"][i], 6)}')
            parameter_rows.append(row)
        frame_info = {frame["id"]: frame for frame in ds["frames"]}
        views = {name: {v["id"]: v["metrics"] for v in ds["results"][name].get("per_view", [])} for name in names}
        frame_table = []
        for frame in ds["input_frames"]:
            frame_id = frame["id"]
            preview = frame_info.get(frame_id, {}).get("preview")
            row = [link(prefix + preview, frame_id) if preview else text(frame_id),
                   "标定" if frame["role"] == "calibration" else "验证"]
            for name in names:
                metrics = views[name].get(frame_id)
                label = f'{_number(metrics["mre_px"], 6)} / {_number(metrics["rmse_px"], 6)}' if metrics else "—"
                row.append(label)
            frame_table.append(row)
        records = []
        for name in names:
            result = ds["results"][name]
            records.append([text(name), link(f'{prefix}methods/{name}/calibration.json', 'calibration.json')
                            if result["status"] == "completed" else '—'])
        sections.append(f'<section><h2>{text(camera["serial"])} · 结果详情</h2>'
            '<h3>角点坐标差异</h3>' + table(["方案", "参照", "帧数", "均值 (px)", "RMSE (px)", "最大 (px)", "集合状态"], differences) +
            '<p class="caption">差异 = 方案 − 参照，非真值误差。</p>'
            f'<details><summary>逐图 MRE / RMSE（px）与角点图</summary><p class="caption">蓝圈：'
            f'{text(ds.get("baseline", names[0]))}；橙十字：当前方案。点击帧 ID 查看图片。</p>' +
            table(["原帧 ID", "用途", *names], frame_table) + '</details>'
            '<details><summary>内参、畸变及估计标准差</summary>' + table(["参数（估计值 ± 标准差）", *names], parameter_rows) +
            '<p class="caption">焦距与主点：px；D 无量纲。</p></details>'
            '<details><summary>标定与诊断 JSON</summary>' + table(["方案", "标定记录"], records) + '</details></section>')
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>检测器 Compare 实验</title><style>{STYLE}</style></head><body><main>
<header><h1>检测器对比</h1><p class="intro">配置与输入：{link("input_manifest.json", "input_manifest.json")}</p>
{table(["设备", "方案", "后端", "角点处理", "检测成功", "配置 JSON"], algorithms)}</header>
<section><h2>标定与独立验证误差</h2>{table(["设备", "方案", "标定 MRE (px)", "标定 RMSE (px)", "最大误差 (px)", "验证 MRE (px)", "验证 RMSE (px)", "验证状态"], results_rows)}</section>
{"".join(sections)}<footer>{link("comparison.json", "comparison.json")} ·
{link("per_view_comparison.csv", "per_view_comparison.csv")} · {link("corner_differences.csv", "corner_differences.csv")}</footer></main></body></html>'''


def _role(view):
    return "验证" if view["role"] == "validation" else "标定"


def _preview(view):
    return f"previews/{escape(view['id'], quote=True)}.png"


def _review_reason(view, threshold):
    status = view.get("review_status")
    if status == "needs_review":
        return f"原始 RMS {_number(view['rms_px'])} px 超过已有复查阈值 {_number(threshold)} px"
    if status == "not_checked":
        return "没有有效的重投影误差值，尚未检查"
    return "未触发已有复查阈值；不代表人工检查通过"


def _warning_text(message):
    translations = {
        "Weak orientation diversity: add views tilted about both board axes": "姿态多样性不足，建议增加绕标定板两个轴倾斜的图像。",
        "Principal point lies outside the image; inspect calibration": "主点位于图像范围之外，请复查标定结果。",
        "Fewer than five held-out views": "独立验证图像少于 5 张。",
    }
    if message in translations:
        return translations[message]
    coverage = re.fullmatch(r"Limited image coverage \((\d+)/(\d+) cells\); add edge/corner views", message)
    if coverage:
        return f"图像覆盖不足（{coverage[1]}/{coverage[2]} 个网格），建议补充边缘和四角视角。"
    review = re.fullmatch(r"(\d+) views exceed review RMS (.+) px; inspect before excluding", message)
    if review:
        return f"{review[1]} 张标定图像的原始 RMS 超过 {review[2]} px，请先复查，再决定是否排除。"
    return "存在其他诊断信息，请展开原始诊断记录查看。"


def _validation_summary(result, validation):
    status = result.get("validation_status", "evaluated" if validation else "not_run")
    if validation and status == "evaluated":
        return "已完成验证", f"{len(validation)} 张 · RMSE {_number(result['validation']['metrics']['rmse_px'])} px"
    notes = {
        "skipped_off": "本次主动关闭独立验证。",
        "skipped_no_data": "没有独立验证图像。",
        "missing_required": "要求独立验证，但没有验证图像。",
        "failed": "独立验证失败，原因见原始诊断记录。",
        "not_run": "尚未执行独立验证。",
    }
    label = "验证失败" if status == "failed" else "未验证"
    return label, notes.get(status, "没有可用的独立验证结果。")


def _statistics(result, validation, detailed=False):
    columns = ([("rms_px", "原始 RMS（px）"), ("std_px", "标准差（px）"), ("variance_px2", "方差（px²）")]
               if detailed else [("mre_px", "MRE（px）"), ("rmse_px", "RMSE（px）"), ("max_error_px", "最大误差（px）")])
    rows = []
    for label, group in (("标定", result), ("独立验证" if validation else "独立验证（未验证）", result.get("validation", {}))):
        metrics = group.get("metrics") or {}
        cells = "".join(f"<td>{_number(group.get(key) if key == 'rms_px' else metrics.get(key))}</td>" for key, _ in columns)
        rows.append(f"<tr><td>{label}</td>{cells}</tr>")
    headers = "".join(f"<th scope='col'>{label}</th>" for _, label in columns)
    return f"<div class='table-wrap'><table><thead><tr><th scope='col'>数据集</th>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def _review_card(view, threshold):
    metrics = view["metrics"]
    warning = view.get("review_status") == "needs_review"
    return f"""<article class="review-card" data-frame-id="{escape(view['id'], quote=True)}" data-rmse="{metrics['rmse_px']}">
<a href="{_preview(view)}" target="_blank" rel="noopener" aria-label="查看叠加图 {escape(view['id'], quote=True)}"><img src="{_preview(view)}" loading="lazy" alt="{_role(view)}图像 {escape(view['id'], quote=True)} 的角点叠加预览"></a>
<div class="review-body"><a class="frame-id" href="{_preview(view)}" target="_blank" rel="noopener">{escape(view['id'])}</a>
<span class="badge{' warning' if warning else ''}">{_role(view)} · {'需复查' if warning else '误差排序靠前'}</span>
<p class="review-values">RMSE <strong>{_number(metrics['rmse_px'])} px</strong> · 最大误差 {_number(metrics['max_error_px'])} px</p>
<p class="reason">{escape(_review_reason(view, threshold))}</p></div></article>"""


def _view_table(views, threshold):
    rows = []
    labels = {"needs_review": "需复查", "not_flagged": "未触发告警", "not_checked": "未检查"}
    for view in views:
        metrics = view["metrics"]
        cells = "".join(f"<td>{_number(metrics.get(key))}</td>" for key in
                        ("mre_px", "rmse_px", "max_error_px", "std_px", "variance_px2"))
        reason = escape(_review_reason(view, threshold), quote=True)
        rows.append(f'<tr data-frame-id="{escape(view["id"], quote=True)}" data-rmse="{metrics["rmse_px"]}">'
                    f'<td><a class="frame-id" href="{_preview(view)}" target="_blank" rel="noopener">{escape(view["id"])}</a></td>'
                    f'<td>{_role(view)}</td>{cells}<td>{_number(view["rms_px"])}</td>'
                    f'<td title="{reason}">{labels.get(view.get("review_status"), "未检查")}</td></tr>')
    return '<div class="scroll-table"><table id="view-results"><thead><tr><th>帧 ID / 叠加图</th><th>用途</th><th>MRE（px）</th><th>RMSE（px）</th><th>最大误差（px）</th><th>标准差（px）</th><th>方差（px²）</th><th>原始 RMS（px）</th><th>复查状态</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'


def _parameters(result):
    K, D = result["K"], result["D"]
    std = dict(zip(result["std_order"], result["std_intrinsics"]))
    matrix = "".join("<tr>" + "".join(f"<td>{_number(value, 6)}</td>" for value in row) + "</tr>" for row in K)
    intrinsics = "".join(f"<tr><td>{name}</td><td>{_number(K[i][j], 6)}</td><td>{_number(std.get(name), 6)}</td></tr>"
                         for name, i, j in (("fx", 0, 0), ("fy", 1, 1), ("cx", 0, 2), ("cy", 1, 2)))
    distortion = "".join(f"<tr><td>{escape(name)}</td><td>{_number(value, 6)}</td><td>{_number(std.get(name), 6)}</td></tr>"
                         for name, value in zip(result["distortion_order"], D))
    return f"""<div class="parameters"><div><h3>内参矩阵 K</h3>
<div class="table-wrap"><table class="matrix" aria-label="内参矩阵 K"><tbody>{matrix}</tbody></table></div>
<div class="table-wrap"><table><thead><tr><th>参数</th><th>估计值（px）</th><th>估计标准差（px）</th></tr></thead><tbody>{intrinsics}</tbody></table></div></div>
<div><h3>畸变系数 D</h3><div class="table-wrap"><table><thead><tr><th>参数</th><th>估计值（无量纲）</th><th>估计标准差（无量纲）</th></tr></thead><tbody>{distortion}</tbody></table></div>
<p class="caption">k1、k2、k3 为径向畸变；p1、p2 为切向畸变。</p></div></div>
<p class="muted parameters-note">标准差是当前模型内的参数估计不确定性，不包含标定板制造与平整度误差。页面显示值经过格式化，JSON 保存精度保持不变。</p>"""


def render_report_html(result: dict, plots: dict, worst: dict | None) -> str:
    """Render prepared statistics without changing the supplied result or view order."""
    training = result["views"]
    validation = result.get("validation", {}).get("views", [])
    views = sorted(training + validation, key=lambda view: (-view["metrics"]["rmse_px"], view["id"]))
    flagged = [view for view in views if view.get("review_status") == "needs_review"]
    review = flagged or views[:3]
    threshold = result["config"]["calibration"]["review_rms_px"]
    metrics = result["metrics"]
    status, validation_note = _validation_summary(result, validation)
    diagnostics = result.get("diagnostics", {})
    warnings = result.get("warnings", []) + result.get("validation", {}).get("warnings", [])
    warning_html = "".join(f"<li>{escape(_warning_text(message))}</li>" for message in warnings)
    warning_html = warning_html or "<li>未触发已有诊断提示；仍需结合图像及独立验证复查。</li>"
    raw_warnings = "".join(f"<li>{escape(message)}</li>" for message in warnings) or "<li>无</li>"
    raw_reason = escape(result.get("validation_reason", "未记录"))
    raw_reviews = "".join(f"<li><span class='frame-id'>{escape(view['id'])}</span>：{escape(view.get('review_reason', '未记录'))}</li>" for view in views)
    coverage = (f"角点覆盖：{diagnostics['coverage_cells']}/{diagnostics['coverage_total']} 个网格。"
                if "coverage_cells" in diagnostics and "coverage_total" in diagnostics else "")
    sigma = dict(zip(result["std_order"], result["std_intrinsics"]))
    uncertainty = f"焦距估计标准差：σ(fx) = {_number(sigma.get('fx'), 6)} px，σ(fy) = {_number(sigma.get('fy'), 6)} px。"
    cards = "".join(_review_card(view, threshold) for view in review[:3])
    additional = (f'<details><summary>其余 {len(review)-3} 张需复查图像</summary><div class="review-grid">'
                  + ''.join(_review_card(view, threshold) for view in review[3:]) + '</div></details>') if len(review) > 3 else ""
    review_note = (f"共 {len(flagged)} 张图像需复查，按 RMSE 降序排列；标记沿用已有原始 RMS 阈值 {threshold:g} px。"
                   if flagged else "未触发已有复查阈值；以下列出 RMSE 最高的图像供核对，不代表人工检查通过。")
    worst_html = ""
    if worst:
        worst_html = (f'<p class="worst">标定集最大误差角点：<a class="frame-id" href="previews/{escape(worst["id"], quote=True)}.png" target="_blank" rel="noopener">'
                      f'{escape(worst["id"])}</a>，角点 {worst["corner"]}（从 0 开始），'
                      f'坐标 ({worst["point"][0]:.2f}, {worst["point"][1]:.2f}) px，误差 {_number(worst["error_px"])} px；叠加图中以橙圈标出。</p>')
    validation_plots = ""
    if validation:
        validation_plots = f"""<details><summary>独立验证的角点覆盖与残差</summary><div class="charts">
<figure><h3>验证角点覆盖</h3>{plots['validation_coverage']}<figcaption>绿色点为验证图像的观测角点。</figcaption></figure>
<figure><h3>验证残差矢量</h3>{plots['validation_residuals']}<figcaption>固定内参，仅拟合各图标定板位姿；红线放大 30 倍。</figcaption></figure></div></details>"""
    excluded = escape(", ".join(result["excluded_frames"])) or "无"
    model = "针孔模型 · Brown 五参数" if result["model"] == "opencv_pinhole_brown5" else escape(result["model"])
    width, height = result["image_size"]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>相机内参标定报告 · {escape(str(result['camera']['serial']))}</title><style>{STYLE}</style></head>
<body><main><header id="overview"><h1>相机内参标定报告</h1>
<p class="intro muted">相机序列号 <strong>{escape(str(result['camera']['serial']))}</strong> · 分辨率 {width} × {height} px · {model} · 标定图像 {len(training)} 张</p>
<div class="cards"><dl class="metric"><dt>整体 RMSE · 标定集</dt><dd>{_number(metrics['rmse_px'])} <small>px</small></dd></dl>
<dl class="metric"><dt>最大角点误差 · 标定集</dt><dd>{_number(metrics['max_error_px'])} <small>px</small></dd></dl>
<dl class="metric{' attention' if flagged else ''}"><dt>需复查图像 · 标定与验证</dt><dd>{len(flagged)} <small>张</small></dd></dl>
<dl class="metric status{' attention' if status != '已完成验证' else ''}"><dt>独立验证状态</dt><dd>{status}</dd><dd class="muted" style="font-size:13px;font-weight:400">{validation_note}</dd></dl></div>
<p class="note">{('未验证，当前结果不能据此视为标定合格。' if status == '未验证' else '独立验证状态不等于测量精度合格判定。')} 像素重投影误差不等于毫米测量精度；使用时保持镜头、对焦、分辨率与感兴趣区域（ROI）一致。</p></header>

<section id="quality"><h2>第一部分 · 标定质量分析</h2>{_statistics(result, validation)}
<p class="caption">MRE 为平均重投影误差，RMSE 为均方根误差；整体统计汇总全部角点的误差模长。</p>
<details><summary>详细误差统计</summary>{_statistics(result, validation, detailed=True)}
<p class="caption">标准差与方差针对角点误差模长，采用样本统计 ddof=1，与参数估计标准差不同。保留原始 RMS 以便核对计算输出。</p></details>
<div class="diagnostics"><h3>诊断提示</h3><ul>{warning_html}</ul><p class="muted">{coverage} {uncertainty}完整参数见第四部分。</p>
<details><summary>原始诊断记录</summary><ul>{raw_warnings}</ul><p class="source">独立验证原始记录：{raw_reason}</p><p class="source">已排除帧：{excluded}。未自动删除高误差图像。</p>
<details><summary>逐图复查原始原因</summary><ul>{raw_reviews}</ul></details></details></div>
<div class="charts"><figure><h3>标定角点覆盖</h3>{plots['coverage']}<figcaption>蓝色点为观测角点；网格用于查看画面覆盖。</figcaption></figure>
<figure><h3>标定残差矢量</h3>{plots['residuals']}<figcaption>红线为观测点减投影点，放大 30 倍，仅用于显示。</figcaption></figure></div>{validation_plots}</section>

<section id="review"><h2>第二部分 · 异常图像复查</h2><p class="caption">{review_note}</p>
<div class="review-grid">{cards}</div>{additional}{worst_html}
<p class="caption">点击缩略图查看原有叠加图：绿圈为观测角点，红十字为投影点；验证预览左侧为叠加图，右侧为去畸变图（α = 1，黑边为无效区域）。</p></section>

<section id="per-view"><h2>第三部分 · 逐图结果</h2><details id="all-views"><summary>展开全部 {len(views)} 张图像的误差统计（RMSE 降序）</summary>
<p class="caption">点击帧 ID 查看对应预览。复查状态沿用原阈值；“未触发告警”不代表人工检查通过。</p>{_view_table(views, threshold)}</details></section>

<section id="parameters"><h2>第四部分 · 标定参数</h2><details id="parameter-details"><summary>查看内参矩阵 K、畸变系数 D 及估计标准差</summary>{_parameters(result)}</details></section>
<footer><p class="source">源会话：{escape(result['source_session'])}</p><p>标定结果时间（UTC）：{escape(result['created_utc'])}</p></footer></main></body></html>"""
