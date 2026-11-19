"use strict";

const byId = id => document.getElementById(id);
const samplesNav = byId("samples");
const plot = byId("plot");
const notice = byId("notice");
let chartFrame = 0;
const previous = byId("previous");
const next = byId("next");
const batchForm = byId("batch-form");
const sampleSize = byId("sample-size");
const generate = byId("generate");
const resample = byId("resample");
const videoPanel = byId("video-panel");
const cameras = byId("cameras");

const labels = {
  right_waist: "腰关节",
  right_shoulder: "肩关节",
  right_elbow: "肘关节",
  right_forearm_roll: "前臂旋转",
  right_wrist_angle: "腕关节俯仰",
  right_wrist_rotate: "腕关节旋转",
  right_gripper: "夹爪",
};

let dataset;
let batch;
let selected;
let trajectory;
let pending;
let charts = [];
let renderedFrame = 0;
let videoSession;
let chartHovered = false;
let inferenceState = {running: false, ready: false, states: {}, ids: []};
let modelResult;
let animationStart = 0;
let animationFrame = 0;
let reveal = 1;
let resultLoading = false;

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "服务请求失败。");
  return data;
}

function showError(error, retry) {
  notice.hidden = false;
  notice.textContent = error instanceof TypeError ? "连接验收服务失败。" : error.message;
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = "重试";
  button.addEventListener("click", retry);
  notice.append(button);
}

function saveSelection(replace) {
  const url = new URL(location.href);
  if (selected === undefined) url.searchParams.delete("sample");
  else url.searchParams.set("sample", selected);
  if (replace) history.replaceState({batch}, "", url);
  else history.pushState({batch}, "", url);
}

function changeSample(id) {
  if (!batch?.ids.includes(id) || id === selected) return;
  loadSample(id);
  saveSelection(false);
}

function showBatch() {
  samplesNav.replaceChildren(...(batch?.ids ?? []).map((id, index) => {
    const button = document.createElement("button");
    button.className = "sample";
    button.type = "button";
    button.dataset.id = id;
    const order = document.createElement("span");
    order.className = "sample-index";
    order.setAttribute("aria-hidden", "true");
    order.textContent = String(index + 1).padStart(2, "0");
    const detail = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = `第 ${id} 条`;
    const subtitle = document.createElement("small");
    subtitle.textContent = "Solid + Liquid";
    detail.append(title, subtitle);
    button.append(order, detail);
    button.addEventListener("click", () => changeSample(id));
    return button;
  }));
  byId("sample-count").textContent = batch ? `${batch.ids.length} 条` : "";
  byId("batch-status").textContent = batch
    ? `${batch.ids.length} 条固液配对 · Seed ${batch.seed} · ${batch.generated ? "已生成" : "待生成"}`
    : `${dataset.samples.length} 条配对可抽样 · 初始 Seed ${dataset.defaultSeed}`;
  renderReviews();
}

async function updateBatch(replaceBatch) {
  if (!batchForm.reportValidity() || generate.disabled || inferenceState.running) return;
  const count = sampleSize.valueAsNumber;
  generate.disabled = resample.disabled = sampleSize.disabled = true;
  try {
    const seed = (batch?.seed ?? dataset.defaultSeed) + (batch && replaceBatch ? 1 : 0);
    if (!batch || replaceBatch || batch.ids.length !== count) {
      const params = new URLSearchParams({count, seed});
      if (replaceBatch && batch) params.set("previous", batch.ids.join(","));
      const {ids} = await fetchJSON(`api/batch?${params}`);
      batch = {ids, seed, generated: !replaceBatch};
    } else {
      batch = {...batch, generated: true};
    }
    showBatch();
    refreshInferenceUI();
    loadSample(!replaceBatch && batch.ids.includes(selected) ? selected : batch.ids[0]);
    saveSelection(true);
  } catch (error) {
    byId("batch-status").textContent = error instanceof TypeError ? "连接验收服务失败，请重试。" : error.message;
  } finally {
    generate.disabled = resample.disabled = sampleSize.disabled = inferenceState.running;
  }
}

function restoreBatch() {
  const saved = history.state?.batch;
  if (!Array.isArray(saved?.ids) || !saved.ids.length
      || new Set(saved.ids).size !== saved.ids.length || !Number.isSafeInteger(saved.seed)
      || !saved.ids.every(id => dataset.samples.some(sample => sample.id === id))) return false;
  batch = saved;
  sampleSize.value = batch.ids.length;
  showBatch();
  const requested = Number(new URLSearchParams(location.search).get("sample") ?? NaN);
  loadSample(batch.ids.includes(requested) ? requested : batch.ids[0]);
  return true;
}

async function loadSample(id) {
  pending?.abort();
  const request = new AbortController();
  pending = request;
  selected = id;
  renderReviews();
  modelResult = undefined;
  cancelAnimationFrame(animationFrame);
  byId("animation").disabled = true;
  renderMetrics();
  chartHovered = false;
  trajectory = undefined;
  const index = batch.ids.indexOf(id);
  const sample = dataset.samples.find(sample => sample.id === id);

  for (const button of samplesNav.querySelectorAll("button")) {
    if (Number(button.dataset.id) === id) button.setAttribute("aria-current", "true");
    else button.removeAttribute("aria-current");
  }
  const activeSample = samplesNav.querySelector('[aria-current="true"]');
  const itemBounds = activeSample.getBoundingClientRect();
  const listBounds = samplesNav.getBoundingClientRect();
  if (itemBounds.top < listBounds.top) samplesNav.scrollTop -= listBounds.top - itemBounds.top;
  else if (itemBounds.bottom > listBounds.bottom) samplesNav.scrollTop += itemBounds.bottom - listBounds.bottom;
  byId("sample-number").textContent = `SAMPLE ${String(index + 1).padStart(2, "0")}`;
  byId("sample-title").textContent = `第 ${id} 条`;
  byId("sample-meta").textContent = `Solid + Liquid · ${sample.frames} 帧 · ${dataset.fps} FPS`;
  byId("chart-title").textContent = `第 ${id} 条 · 7 维轨迹对比图`;
  byId("position").textContent = `${index + 1} / ${batch.ids.length}`;
  previous.disabled = index === 0;
  next.disabled = index === batch.ids.length - 1;
  plot.hidden = true;
  videoPanel.hidden = true;
  byId("playback-controls").hidden = true;
  if (videoSession) {
    videoSession.active = false;
    cancelAnimationFrame(videoSession.animation);
  }
  for (const video of cameras.querySelectorAll("video")) {
    video.pause();
    video.removeAttribute("src");
    video.load();
  }
  cameras.replaceChildren();
  notice.hidden = false;
  if (!batch.generated) {
    notice.textContent = "已换一批数据，点击「生成验收批次」查看轨迹。";
    return;
  }
  notice.textContent = `正在加载第 ${id} 条轨迹…`;
  try {
    const data = await fetchJSON(`api/samples/${id}`, {signal: request.signal});
    if (request.signal.aborted) return;
    trajectory = data;
    buildCharts();
    plot.hidden = false;
    notice.hidden = true;
    setFrame(0);
    showVideos(id);
    fetchResult();
  } catch (error) {
    if (!request.signal.aborted) showError(error, () => loadSample(id));
  }
}

async function initialize() {
  notice.textContent = "正在加载样本…";
  try {
    dataset = await fetchJSON("api/samples");
    sampleSize.max = dataset.samples.length;
    sampleSize.value = Math.min(Number(sampleSize.value), dataset.samples.length);
    generate.disabled = resample.disabled = false;
    if (!restoreBatch()) {
      showBatch();
      notice.textContent = "选择抽样条数，然后点击「生成验收批次」。";
    }
  } catch (error) {
    showError(error, initialize);
  }
}

function showVideos(id) {
  const names = {cam_left_wrist: "左腕·cam_left_wrist", cam_high: "顶部·cam_high", cam_right_wrist: "右腕·cam_right_wrist"};
  const session = {active: true, videos: [], playing: false, time: 0, animation: 0, starting: false};
  videoSession = session;
  byId("video-play").disabled = true;
  byId("video-play").textContent = "播放";
  byId("video-progress").disabled = true;
  updateVideoProgress(0);
  byId("video-status").textContent = "正在加载三路画面…";
  for (const camera of dataset.cameras) {
    const figure = document.createElement("figure");
    const caption = document.createElement("figcaption");
    caption.textContent = names[camera] || camera;
    const video = document.createElement("video");
    video.playsInline = true;
    video.muted = true;
    video.preload = "auto";
    video.setAttribute("aria-label", caption.textContent + "采集视频");
    session.videos.push(video);
    video.addEventListener("loadedmetadata", () => {
      if (session.active) video.currentTime = session.time;
    });
    for (const event of ["loadeddata", "canplay", "seeked"]) {
      video.addEventListener(event, () => {
        if (session.active) resumeVideos(session);
      });
    }
    video.addEventListener("waiting", () => {
      if (!session.active || !session.playing) return;
      session.time = session.videos[0].currentTime;
      session.videos.forEach(item => item.pause());
      byId("video-status").textContent = "正在缓冲三路视频…";
    });
    video.addEventListener("ended", () => {
      if (session.active) pauseVideos();
    });
    video.addEventListener("error", () => {
      if (!session.active) return;
      pauseVideos();
      byId("video-play").disabled = true;
      byId("video-status").textContent = "视频加载失败，请重新打开样本或使用支持 AV1 的浏览器。";
    });
    video.src = `api/samples/${id}/video/${camera}.mp4`;
    figure.append(caption, video);
    cameras.append(figure);
  }
  videoPanel.hidden = false;
  byId("playback-controls").hidden = false;
}

function updateVideoProgress(time) {
  const index = Math.min(trajectory.frames.length - 1, Math.floor(time * dataset.fps));
  const progress = byId("video-progress");
  progress.value = index;
  const frameText = `帧 ${trajectory.frames[index]} / ${trajectory.frames.length}`;
  byId("video-frame").textContent = frameText;
  progress.setAttribute("aria-valuetext", `${time.toFixed(2)} 秒，${frameText}`);
  byId("video-time").textContent = `${time.toFixed(2)} / ${(trajectory.frames.length / dataset.fps).toFixed(2)} s`;
}

function pauseVideos() {
  const session = videoSession;
  if (!session?.active) return;
  session.playing = false;
  cancelAnimationFrame(session.animation);
  session.videos.forEach(video => video.pause());
  byId("video-play").textContent = "播放";
}

function seekVideos(index) {
  const session = videoSession;
  if (!session?.active) return;
  pauseVideos();
  session.time = trajectory.timestamps[index];
  for (const video of session.videos) {
    if (video.readyState >= 1) video.currentTime = session.time;
  }
  updateVideoProgress(session.time);
  setFrame(index);
  byId("video-status").textContent = "正在定位三路画面…";
}

async function resumeVideos(session) {
  if (!session.active || !session.videos.every(video => video.readyState >= 2 && !video.seeking)) return;
  byId("video-play").disabled = false;
  byId("video-progress").disabled = false;
  if (!session.playing) {
    byId("video-status").textContent = "已暂停";
    return;
  }
  if (session.starting) return;
  session.starting = true;
  try {
    await Promise.all(session.videos.map(video => video.play()));
    if (!session.active || !session.playing) {
      session.videos.forEach(video => video.pause());
      return;
    }
    byId("video-status").textContent = "三路同步播放中";
    cancelAnimationFrame(session.animation);
    session.animation = requestAnimationFrame(() => trackPlayback(session));
  } catch (error) {
    if (session.active && session.playing) {
      pauseVideos();
      byId("video-status").textContent = "播放中断，请点击播放重试。";
    }
  } finally {
    session.starting = false;
  }
}

function trackPlayback(session) {
  if (!session.active || !session.playing) return;
  const master = session.videos[0];
  session.time = master.currentTime;
  for (const video of session.videos.slice(1)) {
    const drift = video.currentTime - session.time;
    if (!video.seeking && Math.abs(drift) > .12) video.currentTime = session.time;
    video.playbackRate = Math.abs(drift) < .025 ? 1 : (drift > 0 ? .97 : 1.03);
  }
  updateVideoProgress(session.time);
  if (!chartHovered) setFrame(Math.min(trajectory.frames.length - 1, Math.floor(session.time * dataset.fps)));
  session.animation = requestAnimationFrame(() => trackPlayback(session));
}

byId("video-play").addEventListener("click", () => {
  const session = videoSession;
  if (!session?.active) return;
  if (session.playing) {
    pauseVideos();
    byId("video-status").textContent = "已暂停";
    return;
  }
  if (session.videos[0].ended) seekVideos(0);
  session.playing = true;
  byId("video-play").textContent = "暂停";
  resumeVideos(session);
});
byId("video-progress").addEventListener("input", event => seekVideos(Number(event.target.value)));

function buildCharts() {
  const frames = trajectory.frames;
  byId("video-progress").max = frames.length - 1;
  charts = dataset.joints.map((joint, dimension) => {
    const figure = document.createElement("figure");
    figure.className = "joint";
    const caption = document.createElement("figcaption");
    const title = document.createElement("strong");
    title.textContent = `J${String(dimension + 1).padStart(2, "0")} · ${labels[joint] || joint}`;
    const unitText = joint === "right_gripper" ? "m" : "rad";
    const unit = document.createElement("span");
    unit.className = "joint-unit";
    unit.textContent = unitText;
    const value = document.createElement("span");
    value.className = "joint-value";
    caption.append(title, unit, value);
    const grid = svgElement("g", {class: "chart-grid"});
    const line = svgElement("path", {class: "chart-line"});
    const cursor = svgElement("line", {class: "chart-cursor"});
    const marker = svgElement("circle", {class: "chart-marker", r: 3.2});
    const svg = svgElement("svg", {
      role: "img",
      "aria-label": `${title.textContent}参考轨迹，悬停查看数值，点击定位视频`,
    });
    const modelLine = svgElement("path", {class: "model-line"});
    svg.append(grid, line, modelLine, cursor, marker);
    figure.append(caption, svg);

    const series = trajectory.reference[dimension];
    let min = Infinity;
    let max = -Infinity;
    for (const point of [...series, ...(modelResult?.prediction[dimension] ?? [])]) {
      if (point === null) continue;
      min = Math.min(min, point);
      max = Math.max(max, point);
    }
    if (min !== Infinity) {
      const padding = Math.max((max - min) * .08, .00001);
      min -= padding;
      max += padding;
    }
    svg.addEventListener("pointermove", event => {
      if (event.pointerType === "mouse") {
        chartHovered = true;
        locatePointer(event, svg);
      }
    });
    svg.addEventListener("pointerleave", () => { chartHovered = false; });
    svg.addEventListener("click", event => {
      locatePointer(event, svg);
      seekVideos(chartFrame);
    });
    return {figure, svg, grid, line, modelLine, cursor, marker, value, series, min, max};
  });
  byId("charts").replaceChildren(...charts.map(chart => chart.figure));
}

function locatePointer(event, svg) {
  const rect = svg.getBoundingClientRect();
  const frames = trajectory.frames;
  const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left - 52) / (rect.width - 66)));
  const target = frames[0] + (frames[frames.length - 1] - frames[0]) * fraction;
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (frames[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && target - frames[lo - 1] < frames[lo] - target) lo--;
  setFrame(lo);
}

function setFrame(index) {
  chartFrame = index;
  const phase = index < trajectory.splitFrame ? "固体" : "液体";
  const frameText = `${phase} · 帧 ${trajectory.frames[index]} · ${trajectory.timestamps[index].toFixed(2)} s`;
  byId("frame-value").textContent = frameText;
  if (!renderedFrame) renderedFrame = requestAnimationFrame(drawCharts);
}

function svgElement(name, attributes) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  setAttributes(element, attributes);
  return element;
}

function setAttributes(element, attributes) {
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
}

function drawCharts() {
  renderedFrame = 0;
  if (!trajectory || plot.hidden) return;
  const index = chartFrame;
  for (const [dimension, chart] of charts.entries()) {
    drawChart(chart, index, dimension === charts.length - 1);
  }
}

function drawChart(chart, index, showFrameTicks) {
  const {svg, cursor, marker, value, series, min} = chart;
  const current = series[index];
  value.textContent = current === null ? "—" : current.toFixed(5);
  const {width, height} = svg.getBoundingClientRect();
  if (width < 90 || !height) return;
  if (width !== chart.width || height !== chart.height) {
    chart.width = width;
    chart.height = height;
    layoutChart(chart, showFrameTicks);
  }
  if (min === Infinity) return;
  if (modelResult) {
    const points = modelResult.prediction[charts.indexOf(chart)];
    const count = Math.floor(points.length * reveal);
    chart.modelLine.setAttribute("d", points.slice(0, count).map((point, offset) =>
      `${offset === 0 || offset === trajectory.splitFrame ? "M" : "L"}${chart.x(trajectory.frames[offset]).toFixed(2)} ${chart.y(point).toFixed(2)}`).join(" "));
  }
  const position = chart.x(trajectory.frames[index]);
  setAttributes(cursor, {x1: position, x2: position});
  marker.style.display = current === null ? "none" : "";
  if (current !== null) setAttributes(marker, {cx: position, cy: chart.y(current)});
}

function layoutChart(chart, showFrameTicks) {
  const {svg, grid, line, cursor, marker, width, height, min, max, series} = chart;
  const left = 52;
  const right = width - 14;
  const top = 7;
  const bottom = height - (showFrameTicks ? 17 : 7);
  const frames = trajectory.frames;
  const first = frames[0];
  const last = frames[frames.length - 1];
  const x = frame => left + (right - left) * (last === first ? .5 : (frame - first) / (last - first));
  const y = point => bottom - (bottom - top) * (point - min) / (max - min);
  chart.x = x;
  chart.y = y;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  grid.replaceChildren();

  function tickLabel(text, x, y, anchor) {
    const label = svgElement("text", {x, y, "text-anchor": anchor});
    label.textContent = text;
    grid.append(label);
  }

  if (min === Infinity) {
    tickLabel("该维度没有有效数值", left, height / 2, "start");
    cursor.style.display = "none";
    marker.style.display = "none";
    return;
  }

  for (const tick of [min, (min + max) / 2, max]) {
    const position = y(tick);
    grid.append(svgElement("line", {x1: left, y1: position, x2: right, y2: position}));
    tickLabel(tick.toFixed(max - min < .02 ? 4 : 2), left - 6, position, "end");
  }
  const tickCount = Math.min(last - first, width > 440 ? 4 : 2);
  for (let i = 0; i <= tickCount; i++) {
    const tick = first + (tickCount ? Math.round((last - first) * i / tickCount) : 0);
    const position = x(tick);
    grid.append(svgElement("line", {x1: position, y1: top, x2: position, y2: bottom}));
    if (showFrameTicks) tickLabel(tick, Math.min(position, width - 18), bottom + 9, "middle");
  }

  const split = x(trajectory.splitFrame);
  grid.append(svgElement("line", {class: "chart-split", x1: split, x2: split, y1: top, y2: bottom}));
  let connected = false;
  const commands = [];
  for (const [offset, point] of series.entries()) {
    if (point === null) {
      connected = false;
      continue;
    }
    commands.push(`${connected && offset !== trajectory.splitFrame ? "L" : "M"}${x(frames[offset]).toFixed(2)} ${y(point).toFixed(2)}`);
    connected = true;
  }
  line.setAttribute("d", commands.join(" "));
  setAttributes(cursor, {y1: top, y2: bottom});
}

window.addEventListener("popstate", () => {
  if (dataset) restoreBatch();
});
batchForm.addEventListener("submit", event => {
  event.preventDefault();
  updateBatch(false);
});
resample.addEventListener("click", () => updateBatch(true));
previous.addEventListener("click", () => changeSample(batch.ids[batch.ids.indexOf(selected) - 1]));
next.addEventListener("click", () => changeSample(batch.ids[batch.ids.indexOf(selected) + 1]));
new ResizeObserver(drawCharts).observe(plot);

const dialog = byId("exit-dialog");
const confirmExit = byId("confirm-exit");
const cancelExit = byId("cancel-exit");
const exitError = byId("exit-error");

byId("exit").addEventListener("click", () => {
  exitError.hidden = true;
  dialog.showModal();
});
dialog.addEventListener("cancel", event => {
  if (confirmExit.disabled) event.preventDefault();
});
confirmExit.addEventListener("click", async () => {
  confirmExit.disabled = true;
  cancelExit.disabled = true;
  exitError.hidden = true;
  try {
    await fetchJSON("api/shutdown", {
      method: "POST",
      headers: {"X-Exit-Token": document.querySelector('meta[name="exit-token"]').content},
    });
    pending?.abort();
    pauseVideos();
    dialog.close();
    byId("workspace").hidden = true;
    byId("closed").hidden = false;
    byId("closed-title").focus();
  } catch (error) {
    exitError.textContent = error instanceof TypeError ? "无法连接验收服务，请重试。" : error.message;
    exitError.hidden = false;
    confirmExit.disabled = false;
    cancelExit.disabled = false;
    confirmExit.focus();
  }
});


function refreshInferenceUI() {
  const matching = batch && JSON.stringify(batch.ids) === JSON.stringify(inferenceState.ids);
  byId("model-status").replaceChildren();
  const lamp = document.createElement("span");
  lamp.className = "lamp " + (inferenceState.ready ? "completed" : "waiting");
  byId("model-status").append(lamp, inferenceState.ready ? "模型推理服务已就绪" : "等待推理模型服务");
  generate.disabled = resample.disabled = sampleSize.disabled = inferenceState.running;
  byId("start-inference").disabled = !batch?.generated || inferenceState.running || !inferenceState.ready;
  byId("stop-inference").disabled = !inferenceState.running || inferenceState.stopping;
  byId("inference-status").textContent = inferenceState.error ||
    (inferenceState.stopping ? "正在停止推理…" : inferenceState.running ? "正在按样本顺序推理…" : "");
  for (const button of samplesNav.querySelectorAll("button")) {
    let statusGroup = button.querySelector(".sample-status");
    if (!statusGroup) {
      statusGroup = document.createElement("span");
      statusGroup.className = "sample-status";
      const indicator = document.createElement("span");
      indicator.setAttribute("aria-hidden", "true");
      const label = document.createElement("span");
      statusGroup.append(indicator, label);
      button.append(statusGroup);
    }
    const status = matching ? inferenceState.states[button.dataset.id] : "pending";
    statusGroup.firstElementChild.className = "lamp " + (status || "pending");
    statusGroup.lastElementChild.textContent =
      {completed: "已完成推理", running: "推理中"}[status] || "等待推理中";
  }
}

function renderMetrics(result) {
  byId("rmse-overall").textContent = result ? result.overall.toFixed(6) : "—";
  byId("rmse-status").textContent = result ? "J01–J06 RMSE 的均方根" : "等待本条推理完成";
  for (let i = 0; i < 7; i++) {
    byId(`rmse-j${i + 1}`).textContent = result ? result.rmse[i].toFixed(6) : "—";
  }
}

async function fetchResult() {
  if (!trajectory || modelResult || resultLoading
      || JSON.stringify(batch?.ids) !== JSON.stringify(inferenceState.ids)
      || inferenceState.states[selected] !== "completed") return;
  const episode = selected;
  resultLoading = true;
  try {
    const result = await fetchJSON(`api/inference/${episode}`);
    if (selected !== episode || !trajectory || !result) return;
    modelResult = result;
    renderMetrics(result);
    buildCharts();
    byId("animation").disabled = false;
    replayAnimation();
  } finally {
    resultLoading = false;
  }
}

function replayAnimation() {
  cancelAnimationFrame(animationFrame);
  animationStart = performance.now();
  reveal = matchMedia("(prefers-reduced-motion: reduce)").matches ? 1 : 0;
  function tick(now) {
    reveal = Math.min(1, (now - animationStart) / 5000);
    drawCharts();
    byId("animation").textContent = reveal < 1 ? "跳过动画" : "重播";
    if (reveal < 1) animationFrame = requestAnimationFrame(tick);
  }
  if (reveal === 1) drawCharts();
  else animationFrame = requestAnimationFrame(tick);
}
byId("animation").addEventListener("click", () => {
  if (reveal < 1) {
    cancelAnimationFrame(animationFrame);
    reveal = 1;
    drawCharts();
    byId("animation").textContent = "重播";
  } else replayAnimation();
});
async function inferenceCommand(command) {
  try {
    inferenceState = await fetchJSON(`api/inference/${command}`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ids: batch?.ids}),
    });
    refreshInferenceUI();
  } catch (error) {
    byId("inference-status").textContent = error.message;
  }
}
byId("start-inference").addEventListener("click", () => inferenceCommand("start"));
byId("stop-inference").addEventListener("click", () => inferenceCommand("stop"));
async function pollInference() {
  try {
    inferenceState = await fetchJSON("api/inference");
    if (dataset) refreshInferenceUI();
    await fetchResult();
  } catch (error) {
    byId("inference-status").textContent = "推理状态连接中断，正在重试";
  }
  if (!byId("workspace").hidden) setTimeout(pollInference, 1000);
}
pollInference();

let reviewStore = {};
try {
  reviewStore = JSON.parse(localStorage.getItem("acceptance-reviews-v1") || "{}");
  if (!reviewStore || typeof reviewStore !== "object" || Array.isArray(reviewStore)) reviewStore = {};
} catch (error) {
  byId("review-error").textContent = "读取本地评定记录失败，请检查浏览器存储设置。";
}

function reviewKey() {
  return JSON.stringify([batch.seed, batch.ids]);
}

function currentReview() {
  const key = reviewKey();
  if (!reviewStore[key] || typeof reviewStore[key] !== "object") {
    reviewStore[key] = {items: {}, notes: ""};
  }
  if (!reviewStore[key].items || typeof reviewStore[key].items !== "object") reviewStore[key].items = {};
  return reviewStore[key];
}

function persistReviews() {
  try {
    localStorage.setItem("acceptance-reviews-v1", JSON.stringify(reviewStore));
    byId("review-error").textContent = "";
    return true;
  } catch (error) {
    byId("review-error").textContent = "本地保存失败，请检查浏览器存储空间和权限。";
    return false;
  }
}

function renderReviews() {
  const enabled = !!batch?.generated;
  const record = batch ? currentReview() : {items: {}};
  const current = record.items[selected] || {};
  byId("notes-open").disabled = !batch;
  for (const id of ["mark-success", "mark-failure"]) byId(id).disabled = !enabled || !!current.confirmed;
  byId("mark-success").setAttribute("aria-pressed", current.choice === "success");
  byId("mark-failure").setAttribute("aria-pressed", current.choice === "failure");
  byId("confirm-review").disabled = !enabled || !!current.confirmed || !current.choice;
  byId("reopen-review").disabled = !enabled || !current.confirmed;
  const index = batch?.ids.indexOf(selected) ?? -1;
  byId("review-previous").disabled = index <= 0;
  byId("review-next").disabled = index < 0 || index === batch.ids.length - 1;
  let success = 0;
  let confirmed = 0;
  byId("outcome-grid").replaceChildren(...(batch?.ids ?? []).map((id, position) => {
    const item = record.items[id] || {};
    if (item.confirmed) {
      confirmed++;
      if (item.choice === "success") success++;
    }
    const button = document.createElement("button");
    const result = item.confirmed ? item.choice : "";
    button.type = "button";
    button.className = "outcome-cell " + result;
    if (selected === id) button.setAttribute("aria-current", "true");
    const label = result === "success" ? "成功" : result === "failure" ? "失败" : "未确认";
    button.setAttribute("aria-label", `样本 ${position + 1}，第 ${id} 条，${label}`);
    const symbol = document.createElement("span");
    symbol.className = "outcome-symbol";
    symbol.textContent = result === "success" ? "✓" : result === "failure" ? "✕" : "";
    symbol.setAttribute("aria-hidden", "true");
    const number = document.createElement("span");
    number.textContent = position + 1;
    button.append(symbol, number);
    button.addEventListener("click", () => changeSample(id));
    return button;
  }));
  const count = batch?.ids.length ?? 0;
  byId("success-rate").textContent = `${count ? (100 * success / count).toFixed(1) : "0.0"}%`;
  byId("outcome-count").textContent = `成功 ${success} 条 · 已评定 ${confirmed} / ${count} 条`;
  byId("outcome-complete").hidden = count === 0 || confirmed !== count;
}

for (const choice of ["success", "failure"]) {
  byId("mark-" + choice).addEventListener("click", () => {
    currentReview().items[selected] = {choice, confirmed: false};
    renderReviews();
  });
}
byId("confirm-review").addEventListener("click", () => {
  const item = currentReview().items[selected];
  if (!item?.choice || item.confirmed) return;
  item.confirmed = true;
  if (!persistReviews()) item.confirmed = false;
  renderReviews();
});
byId("reopen-review").addEventListener("click", () => {
  const item = currentReview().items[selected];
  item.confirmed = false;
  if (!persistReviews()) item.confirmed = true;
  renderReviews();
});
byId("review-previous").addEventListener("click", () => changeSample(batch.ids[batch.ids.indexOf(selected) - 1]));
byId("review-next").addEventListener("click", () => changeSample(batch.ids[batch.ids.indexOf(selected) + 1]));
byId("notes-open").addEventListener("click", () => {
  byId("notes-text").value = currentReview().notes || "";
  byId("notes-status").textContent = "自动保存到此浏览器";
  byId("notes-dialog").showModal();
  byId("notes-text").focus();
});
byId("notes-close").addEventListener("click", () => byId("notes-dialog").close());
byId("notes-text").addEventListener("input", event => {
  currentReview().notes = event.target.value;
  byId("notes-status").textContent = persistReviews() ? "已保存" : "保存失败";
});

initialize();
