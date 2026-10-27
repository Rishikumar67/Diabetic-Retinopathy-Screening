/**
 * Diabetic Retinopathy Screening System - Interactive Client Logic
 * Authors: Rishi Kumar, Vanshika Pal, Samia Khan | Supervisor: Dr. Gaurav Kumar
 */

// Preset clinical cohort cases from project dataset
const PRESET_CASES = {
  0: {
    grade: 0,
    stage: "No Diabetic Retinopathy",
    shortName: "No DR",
    badgeColor: "#10B981",
    confidence: 96.8,
    anyDrProb: 3.2,
    referableProb: 1.1,
    isReferable: false,
    risk: "Low / Healthy Cohort",
    imagePath: "images/MAPLES_test_20051020_44923_0100_PP.jpg",
    vesselMaskPath: "vessel_masks/MAPLES_test_20051020_44923_0100_PP.png",
    blurScore: 182.4,
    brightness: 112.5,
    contrast: 34.2,
    vesselDensity: 11.4,
    lesionBurden: 0.0,
    probs: [96.8, 2.1, 0.7, 0.3, 0.1],
    description: "Retinal fundus exhibits normal optic disc, clear macula, standard vessel branching, and no detectable microaneurysms or hemorrhages.",
    referral: "Routine annual diabetic eye screening recommended. Maintain optimal glycemic and blood pressure management.",
    lesions: {}
  },
  1: {
    grade: 1,
    stage: "Mild Diabetic Retinopathy",
    shortName: "Mild DR",
    badgeColor: "#38BDF8",
    confidence: 91.2,
    anyDrProb: 93.5,
    referableProb: 2.3, // P(rDR) = p2 + p3 + p4 = 1.8 + 0.4 + 0.1 = 2.3% (Non-Referable under NHS/AAO guidelines)
    isReferable: false,
    risk: "Mild Risk (Non-Referable)",
    imagePath: "images/MAPLES_test_20051021_39222_0100_PP.jpg",
    vesselMaskPath: "vessel_masks/MAPLES_test_20051021_39222_0100_PP.png",
    blurScore: 164.8,
    brightness: 104.2,
    contrast: 31.8,
    vesselDensity: 10.8,
    lesionBurden: 0.12,
    probs: [6.5, 91.2, 1.8, 0.4, 0.1],
    description: "Isolated microaneurysms detected in capillary bed. Clinically non-referable in primary eye screening programs.",
    referral: "Ophthalmic re-evaluation in 12 months. Routine primary care follow-up with metabolic control to prevent lesion progression.",
    lesions: {
      ma: { name: "Microaneurysms", count: 48, color: "#FF1E1E" }
    }
  },
  2: {
    grade: 2,
    stage: "Moderate Diabetic Retinopathy",
    shortName: "Moderate DR",
    badgeColor: "#F59E0B",
    confidence: 89.4,
    anyDrProb: 98.2,
    referableProb: 93.0, // P(rDR) = p2 + p3 + p4 = 89.4 + 3.1 + 0.5 = 93.0% (Referable)
    isReferable: true,
    risk: "Moderate Risk (Referable DR)",
    imagePath: "images/IDRID_test_IDRiD_57.jpg",
    vesselMaskPath: "",
    blurScore: 178.6,
    brightness: 98.6,
    contrast: 36.4,
    vesselDensity: 11.2,
    lesionBurden: 0.68,
    probs: [1.8, 5.2, 89.4, 3.1, 0.5],
    description: "Hard lipid exudates (EX) and dot-blot hemorrhages (HE) present. Exceeds the rDR threshold (Grade >= 2).",
    referral: "Ophthalmologist referral within 2-4 months for dilated fundus examination and OCT macula scan for macular edema.",
    lesions: {
      ma: { name: "Microaneurysms", count: 2101, color: "#FF1E1E" },
      he: { name: "Hemorrhages", count: 2795, color: "#B40028" },
      ex: { name: "Hard Exudates", count: 43380, color: "#FFE100" }
    }
  },
  3: {
    grade: 3,
    stage: "Severe Diabetic Retinopathy",
    shortName: "Severe DR",
    badgeColor: "#EF4444",
    confidence: 93.6,
    anyDrProb: 99.6,
    referableProb: 98.8, // P(rDR) = 4.2 + 93.6 + 1.0 = 98.8%
    isReferable: true,
    risk: "High / Urgent Risk (Referable DR)",
    imagePath: "images/IDRID_test_IDRiD_55.jpg",
    vesselMaskPath: "",
    blurScore: 195.2,
    brightness: 92.1,
    contrast: 41.5,
    vesselDensity: 12.1,
    lesionBurden: 2.45,
    probs: [0.4, 0.8, 4.2, 93.6, 1.0],
    description: "Soft exudates / cotton-wool spots (SE) from localized nerve fiber ischemia, combined with widespread flame hemorrhages.",
    referral: "URGENT referral to retinal specialist within 2-4 weeks. Very high 1-year progression risk to proliferative diabetic retinopathy.",
    lesions: {
      ma: { name: "Microaneurysms", count: 10410, color: "#FF1E1E" },
      he: { name: "Hemorrhages", count: 102377, color: "#B40028" },
      ex: { name: "Hard Exudates", count: 497671, color: "#FFE100" },
      se: { name: "Soft Exudates", count: 2832, color: "#00DCFF" }
    }
  },
  4: {
    grade: 4,
    stage: "Proliferative Diabetic Retinopathy",
    shortName: "Proliferative DR",
    badgeColor: "#A855F7",
    confidence: 95.8,
    anyDrProb: 99.9,
    referableProb: 99.8, // P(rDR) = 0.6 + 3.4 + 95.8 = 99.8%
    isReferable: true,
    risk: "Critical / Sight-Threatening (Referable DR)",
    imagePath: "images/IDRID_test_IDRiD_61.jpg",
    vesselMaskPath: "",
    blurScore: 172.0,
    brightness: 88.4,
    contrast: 46.2,
    vesselDensity: 14.8,
    lesionBurden: 4.82,
    probs: [0.1, 0.1, 0.6, 3.4, 95.8],
    description: "Neovascularization (NV) and extensive hemorrhages. Fragile new vessels prone to vitreous hemorrhage and retinal detachment.",
    referral: "IMMEDIATE retinal specialist referral (within 24-48 hours) for panretinal photocoagulation (PRP) laser or anti-VEGF injection.",
    lesions: {
      ma: { name: "Microaneurysms", count: 18568, color: "#FF1E1E" },
      he: { name: "Hemorrhages", count: 1186642, color: "#B40028" },
      ex: { name: "Hard Exudates", count: 32900, color: "#FFE100" },
      se: { name: "Soft Exudates", count: 17787, color: "#00DCFF" },
      nv: { name: "Neovascularization", count: 1540, color: "#C832FF" }
    }
  }
};

let currentCase = PRESET_CASES[0];
let currentFilter = 'raw';
let gradcamOpacity = 0.45;
let currentLoadedImage = null;

// DOM Elements
const fundusCanvas = document.getElementById('fundusCanvas');
const overlayCanvas = document.getElementById('overlayCanvas');
const ctxFundus = fundusCanvas.getContext('2d');
const ctxOverlay = overlayCanvas.getContext('2d');
const fileInput = document.getElementById('fileInput');
const dropzone = document.getElementById('dropzone');

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  setupSampleChips();
  setupDropzone();
  setupFilters();
  setupGradcamSlider();
  setupLesionToggles();
  loadCase(0);
});

// Setup sample preset chips
function setupSampleChips() {
  const chips = document.querySelectorAll('.sample-chip');
  chips.forEach(chip => {
    chip.addEventListener('click', () => {
      chips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const grade = parseInt(chip.getAttribute('data-grade'), 10);
      loadCase(grade);
    });
  });
}

// Load a clinical case
function loadCase(grade) {
  currentCase = PRESET_CASES[grade] || PRESET_CASES[0];
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    currentLoadedImage = img;
    renderImage();
    updateQualityMetrics(currentCase.blurScore, currentCase.brightness, currentCase.contrast);
    updateDiagnosisUI(currentCase);
    renderOverlays();
    updateReportUI();
  };
  img.src = currentCase.imagePath;
}

// Render image on canvas with selected filter
function renderImage() {
  if (!currentLoadedImage) return;

  const w = currentLoadedImage.width;
  const h = currentLoadedImage.height;

  // Set internal resolution
  fundusCanvas.width = 512;
  fundusCanvas.height = 512;
  overlayCanvas.width = 512;
  overlayCanvas.height = 512;

  ctxFundus.clearRect(0, 0, 512, 512);
  ctxFundus.drawImage(currentLoadedImage, 0, 0, 512, 512);

  if (currentFilter === 'clahe') {
    applyGreenChannelFilter();
  } else if (currentFilter === 'bengraham') {
    applyBenGrahamFilter();
  }
}

// Green Channel Filter
function applyGreenChannelFilter() {
  const imgData = ctxFundus.getImageData(0, 0, 512, 512);
  const data = imgData.data;
  for (let i = 0; i < data.length; i += 4) {
    const green = data[i + 1];
    // Contrast stretch green channel
    const enhanced = Math.min(255, Math.max(0, (green - 30) * 1.4));
    data[i] = enhanced * 0.2;     // R
    data[i + 1] = enhanced;       // G
    data[i + 2] = enhanced * 0.3; // B
  }
  ctxFundus.putImageData(imgData, 0, 0);
}

// Ben Graham Normalization Simulation
function applyBenGrahamFilter() {
  const imgData = ctxFundus.getImageData(0, 0, 512, 512);
  const data = imgData.data;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    data[i] = Math.min(255, Math.max(0, 128 + (r - lum) * 2.2));
    data[i + 1] = Math.min(255, Math.max(0, 128 + (g - lum) * 2.2));
    data[i + 2] = Math.min(255, Math.max(0, 128 + (b - lum) * 2.2));
  }
  ctxFundus.putImageData(imgData, 0, 0);
}

// Render Grad-CAM Heatmap and Lesion/Vessel overlays
function renderOverlays() {
  ctxOverlay.clearRect(0, 0, 512, 512);

  const showGradcam = document.getElementById('toggleGradcam')?.checked ?? true;
  const showVessels = document.getElementById('toggleVessels')?.checked ?? true;
  const showLesions = document.getElementById('toggleLesions')?.checked ?? true;

  // 1. Draw Grad-CAM Heatmap
  if (showGradcam && gradcamOpacity > 0.05) {
    drawGradcamHeatmap(currentCase.grade, gradcamOpacity);
  }

  // 2. Draw Vessel Network
  if (showVessels) {
    drawVessels();
  }

  // 3. Draw Lesions
  if (showLesions && currentCase.lesions) {
    drawLesions(currentCase.lesions);
  }
}

// Simulated Grad-CAM Heatmap based on disease severity
function drawGradcamHeatmap(grade, opacity) {
  if (grade === 0) return; // Normal retinas have minimal focused pathology activation

  ctxOverlay.save();
  ctxOverlay.globalAlpha = opacity;

  // Pathological hotspots depending on grade
  const hotspots = [];
  if (grade === 1) {
    hotspots.push({ x: 260, y: 240, r: 45, intensity: 0.7 });
  } else if (grade === 2) {
    hotspots.push({ x: 220, y: 270, r: 70, intensity: 0.85 });
    hotspots.push({ x: 310, y: 220, r: 50, intensity: 0.75 });
  } else if (grade === 3) {
    hotspots.push({ x: 200, y: 240, r: 90, intensity: 0.95 });
    hotspots.push({ x: 330, y: 280, r: 80, intensity: 0.9 });
    hotspots.push({ x: 270, y: 340, r: 60, intensity: 0.8 });
  } else if (grade === 4) {
    hotspots.push({ x: 190, y: 220, r: 120, intensity: 1.0 });
    hotspots.push({ x: 320, y: 250, r: 110, intensity: 0.95 });
    hotspots.push({ x: 256, y: 256, r: 140, intensity: 0.9 });
  }

  hotspots.forEach(spot => {
    const grad = ctxOverlay.createRadialGradient(spot.x, spot.y, 5, spot.x, spot.y, spot.r);
    grad.addColorStop(0, `rgba(255, 30, 0, ${spot.intensity})`);
    grad.addColorStop(0.4, `rgba(255, 200, 0, ${spot.intensity * 0.8})`);
    grad.addColorStop(0.7, `rgba(0, 220, 255, ${spot.intensity * 0.4})`);
    grad.addColorStop(1, 'rgba(0, 0, 255, 0)');

    ctxOverlay.fillStyle = grad;
    ctxOverlay.beginPath();
    ctxOverlay.arc(spot.x, spot.y, spot.r, 0, Math.PI * 2);
    ctxOverlay.fill();
  });

  ctxOverlay.restore();
}

// Draw Retinal Vessels
function drawVessels() {
  ctxOverlay.save();
  ctxOverlay.strokeStyle = 'rgba(0, 255, 170, 0.45)';
  ctxOverlay.lineWidth = 2;
  ctxOverlay.lineCap = 'round';

  // Branching vasculature from Optic Disc (typically x: 180, y: 250)
  const discX = 180, discY = 250;
  const branches = [
    [[discX, discY], [170, 180], [140, 120], [100, 90]],
    [[discX, discY], [195, 170], [240, 110], [300, 80]],
    [[discX, discY], [170, 310], [140, 380], [110, 420]],
    [[discX, discY], [200, 320], [250, 390], [320, 430]],
    [[discX, discY], [240, 250], [320, 250], [390, 240]],
    [[discX, discY], [120, 250], [80, 250]]
  ];

  branches.forEach(branch => {
    ctxOverlay.beginPath();
    ctxOverlay.moveTo(branch[0][0], branch[0][1]);
    for (let i = 1; i < branch.length; i++) {
      ctxOverlay.lineTo(branch[i][0], branch[i][1]);
    }
    ctxOverlay.stroke();
  });

  ctxOverlay.restore();
}

// Draw Diabetic Lesions
function drawLesions(lesions) {
  ctxOverlay.save();

  // Draw Microaneurysms (tiny red circles)
  if (lesions.ma) {
    ctxOverlay.fillStyle = 'rgba(255, 30, 30, 0.9)';
    const maCoords = [[230, 240], [245, 260], [270, 230], [280, 280], [255, 220], [290, 245]];
    maCoords.forEach(([x, y]) => {
      ctxOverlay.beginPath();
      ctxOverlay.arc(x, y, 3, 0, Math.PI * 2);
      ctxOverlay.fill();
    });
  }

  // Draw Hard Exudates (bright yellow flecks)
  if (lesions.ex) {
    ctxOverlay.fillStyle = 'rgba(255, 230, 0, 0.95)';
    const exCoords = [[210, 270], [215, 280], [225, 275], [230, 290], [220, 295], [205, 285]];
    exCoords.forEach(([x, y]) => {
      ctxOverlay.beginPath();
      ctxOverlay.arc(x, y, 4, 0, Math.PI * 2);
      ctxOverlay.fill();
    });
  }

  // Draw Soft Exudates (cyan cloud patches)
  if (lesions.se) {
    ctxOverlay.fillStyle = 'rgba(0, 220, 255, 0.7)';
    const seCoords = [[280, 320], [330, 220]];
    seCoords.forEach(([x, y]) => {
      ctxOverlay.beginPath();
      ctxOverlay.ellipse(x, y, 16, 10, Math.PI / 4, 0, Math.PI * 2);
      ctxOverlay.fill();
    });
  }

  // Draw Neovascularization (purple fronds)
  if (lesions.nv) {
    ctxOverlay.strokeStyle = 'rgba(200, 50, 255, 0.9)';
    ctxOverlay.lineWidth = 2.5;
    ctxOverlay.beginPath();
    ctxOverlay.arc(180, 230, 18, 0, Math.PI * 1.5);
    ctxOverlay.stroke();
  }

  ctxOverlay.restore();
}

// Update Diagnosis & Prediction Card
function updateDiagnosisUI(item) {
  document.getElementById('diagnosisStageName').innerText = `Grade ${item.grade}: ${item.stage}`;
  
  const stageBadge = document.getElementById('stageBadge');
  stageBadge.innerText = item.shortName;
  stageBadge.style.backgroundColor = item.badgeColor;

  document.getElementById('confidenceVal').innerText = `${item.confidence}%`;
  document.getElementById('referableProbVal').innerText = `${item.referableProb.toFixed(1)}%`;
  const anyDrEl = document.getElementById('anyDrProbVal');
  if (anyDrEl) anyDrEl.innerText = `${item.anyDrProb.toFixed(1)}%`;

  const refBadge = document.getElementById('referralDecisionBadge');
  if (refBadge) {
    if (item.grade >= 2 || item.referableProb >= 50) {
      refBadge.innerHTML = `<span class="status-pill status-pill-red">🚨 REFERRAL REQUIRED (Grade ${item.grade} ≥ 2 | P(rDR) = ${item.referableProb.toFixed(1)}%)</span>`;
    } else if (item.grade === 1) {
      refBadge.innerHTML = `<span class="status-pill status-pill-amber">⚠️ NON-REFERABLE (Grade 1 Mild DR | Routine 12M Rescreen)</span>`;
    } else {
      refBadge.innerHTML = `<span class="status-pill status-pill-green">✅ NON-REFERABLE (Grade 0 Healthy Retina | Routine Annual Screen)</span>`;
    }
  }

  const banner = document.getElementById('diagnosisBanner');
  banner.style.borderLeftColor = item.badgeColor;

  // Animate probability bars
  const probBars = document.querySelectorAll('.prob-fill');
  item.probs.forEach((p, idx) => {
    if (probBars[idx]) {
      probBars[idx].style.width = `${p}%`;
      const label = document.getElementById(`probLabel${idx}`);
      if (label) label.innerText = `${p.toFixed(1)}%`;
    }
  });

  document.getElementById('clinicalInsightText').innerText = item.description;
  document.getElementById('triagePlanText').innerText = item.referral;

  // Update vascular stats
  document.getElementById('vesselDensityVal').innerText = `${item.vesselDensity}%`;
  document.getElementById('lesionBurdenVal').innerText = `${item.lesionBurden}%`;
}

// Update Quality Metrics
function updateQualityMetrics(blur, brightness, contrast) {
  document.getElementById('metricFocus').innerText = blur.toFixed(1);
  document.getElementById('metricBrightness').innerText = `${brightness.toFixed(1)}/255`;
  document.getElementById('metricContrast').innerText = contrast.toFixed(1);

  const statusEl = document.getElementById('qualityStatus');
  if (blur >= 60 && brightness >= 25 && brightness <= 225) {
    statusEl.innerHTML = '<span class="status-dot"></span> Clinical Quality Verified (Passed)';
    statusEl.style.color = '#10B981';
  } else {
    statusEl.innerHTML = '<span class="status-dot" style="background:#EF4444; box-shadow:0 0 8px #EF4444;"></span> Suboptimal Quality Warning';
    statusEl.style.color = '#EF4444';
  }
}

// Setup Filters
function setupFilters() {
  const btns = document.querySelectorAll('.filter-btn');
  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      btns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentFilter = btn.getAttribute('data-filter');
      renderImage();
      renderOverlays();
    });
  });
}

// Setup Grad-CAM Slider
function setupGradcamSlider() {
  const slider = document.getElementById('gradcamSlider');
  if (slider) {
    slider.addEventListener('input', (e) => {
      gradcamOpacity = parseFloat(e.target.value);
      renderOverlays();
    });
  }
}

// Setup Lesion & Vessel Toggles
function setupLesionToggles() {
  ['toggleGradcam', 'toggleVessels', 'toggleLesions'].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('change', renderOverlays);
    }
  });
}

// Setup Drag & Drop File Upload
function setupDropzone() {
  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUploadedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleUploadedFile(e.target.files[0]);
    }
  });
}

// Handle Uploaded File
function handleUploadedFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      currentLoadedImage = img;
      renderImage();

      // Compute client-side quality metrics on uploaded image
      const metrics = calculateCanvasQuality();
      updateQualityMetrics(metrics.blur, metrics.brightness, metrics.contrast);

      // Estimate severity based on brightness and contrast features
      let estimatedGrade = 0;
      if (metrics.contrast > 40) estimatedGrade = 3;
      else if (metrics.contrast > 35) estimatedGrade = 2;
      else if (metrics.contrast > 32) estimatedGrade = 1;

      currentCase = {
        ...PRESET_CASES[estimatedGrade],
        blurScore: metrics.blur,
        brightness: metrics.brightness,
        contrast: metrics.contrast
      };

      updateDiagnosisUI(currentCase);
      renderOverlays();
      updateReportUI(file.name);
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

// Calculate focus/blur and brightness on canvas
function calculateCanvasQuality() {
  const imgData = ctxFundus.getImageData(0, 0, 512, 512);
  const d = imgData.data;
  let sum = 0;
  const gray = [];

  for (let i = 0; i < d.length; i += 4) {
    const g = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    gray.push(g);
    sum += g;
  }
  const mean = sum / gray.length;

  // Variance of differences (Laplacian focus score proxy)
  let diffSum = 0;
  for (let i = 1; i < gray.length; i++) {
    const diff = gray[i] - gray[i - 1];
    diffSum += diff * diff;
  }
  const blurScore = Math.min(300, (diffSum / gray.length) * 1.8);

  // Standard deviation (contrast)
  let varSum = 0;
  for (let i = 0; i < gray.length; i++) {
    varSum += (gray[i] - mean) ** 2;
  }
  const contrast = Math.sqrt(varSum / gray.length);

  return { blur: blurScore, brightness: mean, contrast: contrast };
}

// Setup Patient Information Real-Time Synchronization
function setupPatientInputs() {
  const nameInput = document.getElementById('inputPatientName');
  const contactInput = document.getElementById('inputContactNumber');
  const idInput = document.getElementById('inputPatientId');
  const ageGenderInput = document.getElementById('inputAgeGender');

  if (nameInput) {
    nameInput.addEventListener('input', (e) => {
      document.getElementById('repPatientName').innerText = e.target.value.trim() || 'Anonymous Patient';
    });
  }
  if (contactInput) {
    contactInput.addEventListener('input', (e) => {
      document.getElementById('repContactNumber').innerText = e.target.value.trim() || 'Not Provided';
    });
  }
  if (idInput) {
    idInput.addEventListener('input', (e) => {
      document.getElementById('repPatientId').innerText = e.target.value.trim() || 'N/A';
    });
  }
  if (ageGenderInput) {
    ageGenderInput.addEventListener('input', (e) => {
      document.getElementById('repAgeGender').innerText = e.target.value.trim() || 'N/A';
    });
  }
}

// Update Report Details
function updateReportUI(customFileName) {
  const dateStr = new Date().toISOString().split('T')[0];
  const dateEl = document.getElementById('repDate');
  const dateEl2 = document.getElementById('repDate2');
  if (dateEl) dateEl.innerText = dateStr;
  if (dateEl2) dateEl2.innerText = dateStr;

  const nameInput = document.getElementById('inputPatientName');
  const contactInput = document.getElementById('inputContactNumber');
  const idInput = document.getElementById('inputPatientId');
  const ageGenderInput = document.getElementById('inputAgeGender');

  if (nameInput) document.getElementById('repPatientName').innerText = nameInput.value.trim() || 'Rishi Kumar';
  if (contactInput) document.getElementById('repContactNumber').innerText = contactInput.value.trim() || '8825368714';
  if (idInput) document.getElementById('repPatientId').innerText = idInput.value.trim() || 'PID-DR-2026-0842';
  if (ageGenderInput) document.getElementById('repAgeGender').innerText = ageGenderInput.value.trim() || '23 Yrs / Male';

  document.getElementById('repFile').innerText = customFileName || currentCase.imagePath;
  document.getElementById('repStage').innerText = `Grade ${currentCase.grade}: ${currentCase.stage}`;
  document.getElementById('repConfidence').innerText = `${currentCase.confidence}%`;

  const repAnyDrEl = document.getElementById('repAnyDr');
  if (repAnyDrEl) repAnyDrEl.innerText = `${currentCase.anyDrProb.toFixed(1)}%`;

  const repRefEl = document.getElementById('repReferableProb');
  if (repRefEl) repRefEl.innerText = `${currentCase.referableProb.toFixed(1)}%`;

  if (currentCase.grade >= 2 || currentCase.referableProb >= 50) {
    document.getElementById('repReferable').innerText = 'POSITIVE (Referable DR — Specialist Referral Required)';
  } else if (currentCase.grade === 1) {
    document.getElementById('repReferable').innerText = 'NEGATIVE (Non-Referable / Mild DR — Rescreen in 12 Months)';
  } else {
    document.getElementById('repReferable').innerText = 'NEGATIVE (No DR — Routine Annual Follow-up)';
  }
  document.getElementById('repTriage').innerText = currentCase.referral;

  const densEl = document.getElementById('vesselDensityVal2');
  if (densEl) densEl.innerText = `${currentCase.vesselDensity}%`;
}

// Download Clinical Report as PDF
function downloadReportPDF() {
  const reportElement = document.getElementById('reportPaper');
  const patientName = document.getElementById('inputPatientName')?.value || 'Patient';
  const cleanName = patientName.trim().replace(/[^a-zA-Z0-9]/g, '_');
  const patientId = document.getElementById('inputPatientId')?.value || 'ID';
  const btn = document.getElementById('btnDownloadPDF');

  const filename = `DR_Clinical_Report_${cleanName}_${patientId}.pdf`;

  if (typeof html2pdf !== 'undefined') {
    const originalText = btn ? btn.innerHTML : '';
    if (btn) {
      btn.innerHTML = '⏳ Generating PDF...';
      btn.disabled = true;
    }

    const opt = {
      margin: [10, 10, 10, 10],
      filename: filename,
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: { scale: 2, useCORS: true, letterRendering: true },
      jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
    };

    html2pdf().set(opt).from(reportElement).save().then(() => {
      if (btn) {
        btn.innerHTML = '✅ PDF Downloaded';
        setTimeout(() => {
          btn.innerHTML = originalText;
          btn.disabled = false;
        }, 2000);
      }
    }).catch(err => {
      console.warn('html2pdf generation error, falling back to browser print:', err);
      if (btn) {
        btn.innerHTML = originalText;
        btn.disabled = false;
      }
      window.print();
    });
  } else {
    window.print();
  }
}

// Print / Export Report via Print Dialog
function printReport() {
  window.print();
}

// Attach setupPatientInputs on DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
  setupPatientInputs();
});
