const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
const toggleBtn = document.getElementById("toggleBtn");
const restartBtn = document.getElementById("restartBtn");
const speedRange = document.getElementById("speedRange");
const statusText = document.getElementById("statusText");
const progressText = document.getElementById("progressText");
const zoomBtn = document.getElementById("zoomBtn");
const zoomRange = document.getElementById("zoomRange");
const zoomLabel = document.getElementById("zoomLabel");
const fadeBtn = document.getElementById("fadeBtn");

const TAU = Math.PI * 2;
const backgroundColor = "rgba(255, 255, 255, 0.9)";
const circleColor = "rgba(122, 170, 255, 0.75)";
const zoomedCircleColor = "rgba(122, 170, 255, 0.3)";
const arrowColor = "rgba(240, 222, 183, 0.9)";
const zoomedArrowColor = arrowColor;
const traceColor = "rgba(255, 255, 255, 0.95)";
const zoomedTraceColor = traceColor;

const data = window.FLOWER_DATA;
const coefficients = data.coefficients.map(([freq, real, imag]) => ({
  freq,
  real,
  imag,
  radius: Math.hypot(real, imag),
}));

const renderCount = Math.min(data.renderVectors, coefficients.length);
const computeCount = Math.min(data.computeVectors, coefficients.length);

let width = 0;
let height = 0;
let pixelRatio = 1;
let isRunning = true;
let speed = 1;
let startTime = performance.now();
let pausedAt = 0;
let currentT = 0;
let lastSampleT = 0;
let pathPoints = [];
let bgImage = null;
let bgImageReady = false;
let isZooming = false;
let zoomFactor = 2;
let isFadingEnabled = true;
const CONSTANT_SPEED_FACTOR = 1 / 60; // 60 seconds to complete full animation at speed 1
const STEPS_PER_LOOP = 10000;
const FADE_SEGMENTS = 5000; // segments to fade out in front of current position
let currentStep = 0;
let lastSampledStep = 0;

function resizeCanvas() {
  pixelRatio = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  width = Math.floor(window.innerWidth * pixelRatio);
  height = Math.floor(window.innerHeight * pixelRatio);
  canvas.width = width;
  canvas.height = height;
}

function loadBackground() {
  bgImage = new Image();
  bgImage.onload = () => {
    bgImageReady = true;
  };
  bgImage.src = "../output/flower_recolored.png";
}

function getSceneScale() {
  if (!bgImageReady || !bgImage.naturalWidth || !bgImage.naturalHeight) {
    return 1;
  }

  return Math.min(
    width / bgImage.naturalWidth,
    height / bgImage.naturalHeight,
  );
}

function getPathScale(sceneScale) {
  return sceneScale * (data.maxSize / data.drawScale);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function mapElapsedToT(elapsedSeconds) {
  return elapsedSeconds * speed * CONSTANT_SPEED_FACTOR;
}

function toCanvasPoint(x, y, drawScale) {
  return [width * 0.5 + x * drawScale, height * 0.5 - y * drawScale];
}

function evaluateChain(t, count) {
  let x = 0;
  let y = 0;

  const states = [];

  for (let i = 0; i < count; i += 1) {
    const coeff = coefficients[i];
    const angle = TAU * coeff.freq * t;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const dx = coeff.real * cos - coeff.imag * sin;
    const dy = coeff.real * sin + coeff.imag * cos;
    const startX = x;
    const startY = y;

    x += dx;
    y += dy;

    states.push({
      startX,
      startY,
      endX: x,
      endY: y,
      radius: coeff.radius,
    });
  }

  return { x, y, states };
}

function evaluatePoint(t) {
  let x = 0;
  let y = 0;

  for (let i = 0; i < computeCount; i += 1) {
    const coeff = coefficients[i];
    const angle = TAU * coeff.freq * t;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const dx = coeff.real * cos - coeff.imag * sin;
    const dy = coeff.real * sin + coeff.imag * cos;

    x += dx;
    y += dy;
  }

  return { x, y };
}

function samplePath(t) {
  // Convert T to absolute step number (allows multiple loops)
  const nextStep = Math.floor(clamp(t, 0, Infinity) * STEPS_PER_LOOP);

  // Sample all new steps up to nextStep
  for (let step = lastSampledStep; step <= nextStep; step += 1) {
    // Map step back to T in [0,1) range using modulo
    const sampleT = (step % STEPS_PER_LOOP) / STEPS_PER_LOOP;
    const point = evaluatePoint(sampleT);
    const pointIndex = step % STEPS_PER_LOOP;

    // Reuse the same array slot for each loop
    pathPoints[pointIndex] = point;
  }

  currentStep = nextStep;
  lastSampledStep = nextStep;
}

function getBackgroundBounds(sceneScale) {
  if (!bgImageReady) {
    return {
      drawWidth: width * 0.74,
      drawHeight: height * 0.74,
      x: width * 0.13,
      y: height * 0.13,
    };
  }

  const drawWidth = bgImage.naturalWidth * sceneScale;
  const drawHeight = bgImage.naturalHeight * sceneScale;
  const x = (width - drawWidth) * 0.5;
  const y = (height - drawHeight) * 0.5;

  return { drawWidth, drawHeight, x, y };
}

function drawBackground(bounds) {
  if (!bgImageReady) {
    return;
  }

  ctx.save();
  ctx.globalAlpha = 0.55;
  ctx.filter = "saturate(0.8) brightness(0.55) contrast(1.05)";
  ctx.drawImage(bgImage, bounds.x, bounds.y, bounds.drawWidth, bounds.drawHeight);
  ctx.restore();
}

function getPathSegmentAlpha(segmentIndex) {
  if (!isFadingEnabled) {
    return 1.0;
  }

  const currentStepInLoop = currentStep % STEPS_PER_LOOP;
  const hasCompletedFirstLoop = currentStep >= STEPS_PER_LOOP;
  const distanceAhead = (segmentIndex - currentStepInLoop + STEPS_PER_LOOP) % STEPS_PER_LOOP;

  const fadeProgress = Math.min(1, distanceAhead / FADE_SEGMENTS);

  const alpha = Math.pow(0.1, 1 - fadeProgress);

  return alpha;
}

function drawPathSegments(options) {
  const {
    mapPoint,
    strokeStyle,
    lineWidth,
    lineJoin = "round",
    lineCap = "round",
  } = options;

  if (pathPoints.length < 2) {
    return;
  }

  ctx.save();
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = lineJoin;
  ctx.lineCap = lineCap;
  ctx.strokeStyle = strokeStyle;

  for (let i = 0; i < pathPoints.length - 1; i += 1) {
    const point = pathPoints[i];
    const nextPoint = pathPoints[i + 1];
    if (!point || !nextPoint) {
      continue;
    }

    ctx.globalAlpha = getPathSegmentAlpha(i);

    const mappedPoint = mapPoint(point);
    const mappedNextPoint = mapPoint(nextPoint);

    ctx.beginPath();
    ctx.moveTo(mappedPoint[0], mappedPoint[1]);
    ctx.lineTo(mappedNextPoint[0], mappedNextPoint[1]);
    ctx.stroke();

    // Draw connecting segment from last point to first point to close the loop
    if (pathPoints.length > 1 && currentStep >= STEPS_PER_LOOP) {
      const lastPoint = pathPoints[pathPoints.length - 1];
      const firstPoint = pathPoints[0];
      if (lastPoint && firstPoint) {
        ctx.globalAlpha = getPathSegmentAlpha(pathPoints.length - 1);
        const mappedLastPoint = mapPoint(lastPoint);
        const mappedFirstPoint = mapPoint(firstPoint);
        ctx.beginPath();
        ctx.moveTo(mappedLastPoint[0], mappedLastPoint[1]);
        ctx.lineTo(mappedFirstPoint[0], mappedFirstPoint[1]);
        ctx.stroke();
      }
    }
  }

  ctx.restore();
}

function drawChainStates(states, options) {
  const {
    mapPoint,
    radiusScale,
    circleColorValue,
    arrowColorValue,
    circleMinRadius,
    arrowMinRadius,
    circleLineWidth,
    arrowLineWidth,
  } = options;

  for (const state of states) {
    const start = mapPoint(state.startX, state.startY);
    const end = mapPoint(state.endX, state.endY);
    const radius = state.radius * radiusScale;

    if (radius > circleMinRadius) {
      ctx.beginPath();
      ctx.strokeStyle = circleColorValue;
      ctx.lineWidth = circleLineWidth;
      ctx.arc(start[0], start[1], radius, 0, TAU);
      ctx.stroke();
    }

    if (radius > arrowMinRadius) {
      ctx.beginPath();
      ctx.strokeStyle = arrowColorValue;
      ctx.lineWidth = arrowLineWidth;
      ctx.moveTo(start[0], start[1]);
      ctx.lineTo(end[0], end[1]);
      ctx.stroke();
    }
  }
}

function drawPath(scale) {
  drawPathSegments({
    mapPoint: (point) => toCanvasPoint(point.x, point.y, scale),
    strokeStyle: backgroundColor,
    lineWidth: Math.max(1.5, 2.2 * pixelRatio),
    lineJoin: "round",
    lineCap: "butt",
  });
}

function drawChain(t, scale) {
  // Use modulo to map t back to [0, 1)
  const normalizedT = t % 1;
  const { states, x, y } = evaluateChain(normalizedT, renderCount);

  ctx.save();
  drawChainStates(states, {
    mapPoint: (modelX, modelY) => toCanvasPoint(modelX, modelY, scale),
    radiusScale: scale,
    circleColorValue: circleColor,
    arrowColorValue: arrowColor,
    circleMinRadius: 3 * pixelRatio,
    arrowMinRadius: 0.8 * pixelRatio,
    circleLineWidth: Math.max(1, 1.25 * pixelRatio),
    arrowLineWidth: Math.max(1.25, 1.75 * pixelRatio),
  });

  ctx.restore();

  const tipCanvasPoint = toCanvasPoint(x, y, scale);

  return { x: tipCanvasPoint[0], y: tipCanvasPoint[1], modelX: x, modelY: y };
}

function drawCurrentTip(t, scale) {
  // Use modulo to map t back to [0, 1)
  const normalizedT = t % 1;
  const point = evaluatePoint(normalizedT);
  const canvasPoint = toCanvasPoint(point.x, point.y, scale);

  ctx.save();
  ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
  ctx.beginPath();
  ctx.arc(canvasPoint[0], canvasPoint[1], Math.max(2.5, 3.2 * pixelRatio), 0, TAU);
  ctx.fill();
  ctx.restore();
}

function drawZoomRegionIndicator(t, scale) {
  // Use modulo to map t back to [0, 1)
  const normalizedT = t % 1;
  const point = evaluatePoint(normalizedT);
  const canvasPoint = toCanvasPoint(point.x, point.y, scale);

  const zoomBoxSize = (Math.min(width, height) * 0.35) / zoomFactor;
  const halfSize = zoomBoxSize * 0.5;

  ctx.save();
  ctx.strokeStyle = "rgba(138, 179, 255, 0.8)";
  ctx.lineWidth = Math.max(2, 2.5 * pixelRatio);
  ctx.setLineDash([5, 5]);
  ctx.strokeRect(canvasPoint[0] - halfSize, canvasPoint[1] - halfSize, zoomBoxSize, zoomBoxSize);
  ctx.restore();
}

function drawZoomBox(t, scale) {
  // Use modulo to map t back to [0, 1)
  const normalizedT = t % 1;
  const point = evaluatePoint(normalizedT);
  const centerX = point.x;
  const centerY = point.y;

  const zoomBoxSize = Math.min(width, height) * 0.35;
  const marginRight = width * 0.02;
  const zoomBoxX = width - zoomBoxSize - marginRight;
  const zoomBoxY = height * 0.05;

  // Draw zoom box background
  ctx.save();
  ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
  ctx.fillRect(zoomBoxX, zoomBoxY, zoomBoxSize, zoomBoxSize);

  // Draw border
  ctx.strokeStyle = "rgba(138, 179, 255, 0.8)";
  ctx.lineWidth = Math.max(1.5, 2 * pixelRatio);
  ctx.strokeRect(zoomBoxX, zoomBoxY, zoomBoxSize, zoomBoxSize);

  // Draw zoomed content with clipping
  ctx.save();
  ctx.beginPath();
  ctx.rect(zoomBoxX, zoomBoxY, zoomBoxSize, zoomBoxSize);
  ctx.clip();

  ctx.translate(zoomBoxX + zoomBoxSize * 0.5, zoomBoxY + zoomBoxSize * 0.5);

  // Draw zoomed path points with fade effect
  const zoomedScale = scale * zoomFactor;
  drawPathSegments({
    mapPoint: (pathPoint) => [
      (pathPoint.x - centerX) * zoomedScale,
      -(pathPoint.y - centerY) * zoomedScale,
    ],
    strokeStyle: backgroundColor,
    lineWidth: Math.max(1, 1.5 * pixelRatio),
    lineJoin: "round",
    lineCap: "round",
  });

  // Draw zoomed chain (vectors)
  const { states: zoomedStates } = evaluateChain(normalizedT, renderCount);
  drawChainStates(zoomedStates, {
    mapPoint: (modelX, modelY) => [
      (modelX - centerX) * zoomedScale,
      -(modelY - centerY) * zoomedScale,
    ],
    radiusScale: zoomedScale,
    circleColorValue: circleColor,
    arrowColorValue: arrowColor,
    circleMinRadius: 1 * pixelRatio,
    arrowMinRadius: 0.4 * pixelRatio,
    circleLineWidth: Math.max(0.5, 0.75 * pixelRatio),
    arrowLineWidth: Math.max(0.75, 1 * pixelRatio),
  });

  // Draw current tip in zoomed view
  const relTipX = 0;
  const relTipY = 0;
  ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
  ctx.beginPath();
  ctx.arc(relTipX, relTipY, Math.max(1.5, 2 * pixelRatio), 0, TAU);
  ctx.fill();

  ctx.restore();
  ctx.restore();
}

function drawFrame(timestamp) {
  if (!isRunning) {
    return;
  }

  const elapsed = (timestamp - startTime) / 1000;
  // Allow T to exceed 1 for multiple loops
  currentT = mapElapsedToT(elapsed);

  const sceneScale = getSceneScale();
  const backgroundBounds = getBackgroundBounds(sceneScale);
  const drawScale = getPathScale(sceneScale);

  ctx.clearRect(0, 0, width, height);
  drawBackground(backgroundBounds);
  samplePath(currentT);
  drawPath(drawScale);
  drawChain(currentT, drawScale);
  drawCurrentTip(currentT, drawScale);

  if (isZooming) {
    drawZoomRegionIndicator(currentT, drawScale);
    drawZoomBox(currentT, drawScale);
  }

  // Display loop number and percentage within loop
  const loopNumber = Math.floor(currentT);
  const percentInLoop = ((currentT % 1) * 100);
  progressText.textContent = `Loop ${loopNumber + 1}: ${Math.round(percentInLoop)}%`;
  statusText.textContent = "Running";

  // Continue animating indefinitely (or set a max loop limit if desired)
  requestAnimationFrame(drawFrame);
}

function restartAnimation() {
  isRunning = true;
  currentT = 0;
  currentStep = 0;
  lastSampledStep = 0;
  lastSampleT = 0;
  pathPoints = [];
  pausedAt = 0;
  startTime = performance.now();
  toggleBtn.textContent = "Pause";
  statusText.textContent = "Running";
  progressText.textContent = "Loop 1: 0%";
  requestAnimationFrame(drawFrame);
}

toggleBtn.addEventListener("click", () => {
  if (isRunning) {
    isRunning = false;
    pausedAt = performance.now();
    toggleBtn.textContent = "Play";
    statusText.textContent = "Paused";
    return;
  }

  isRunning = true;
  const pauseDuration = performance.now() - pausedAt;
  startTime += pauseDuration;
  toggleBtn.textContent = "Pause";
  statusText.textContent = "Running";
  requestAnimationFrame(drawFrame);
});

restartBtn.addEventListener("click", restartAnimation);
speedRange.addEventListener("input", () => {
  const newSpeed = Number(speedRange.value);
  // Adjust startTime to maintain current T value
  const elapsed = (performance.now() - startTime) / 1000;
  const oldT = mapElapsedToT(elapsed);
  speed = newSpeed;
  // Recalculate startTime so that mapElapsedToT with new speed gives same T
  const newElapsed = oldT / (speed * CONSTANT_SPEED_FACTOR);
  startTime = performance.now() - newElapsed * 1000;
});

zoomBtn.addEventListener("click", () => {
  isZooming = !isZooming;
  zoomLabel.style.display = isZooming ? "grid" : "none";
  zoomBtn.textContent = isZooming ? "Zoom Off" : "Zoom";
});

zoomRange.addEventListener("input", () => {
  zoomFactor = Number(zoomRange.value);
});

fadeBtn.addEventListener("click", () => {
  isFadingEnabled = !isFadingEnabled;
  fadeBtn.textContent = isFadingEnabled ? "Fade Off" : "Fade On";
});

window.addEventListener("resize", () => {
  resizeCanvas();
});

resizeCanvas();
loadBackground();
requestAnimationFrame(drawFrame);