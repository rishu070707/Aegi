/* ════════════════════════════════════════════════
   SENTINEL ALPHA — Core System Logic
   ════════════════════════════════════════════════ */

const Sentinel = {
  streamActive: false,
  statusInterval: null,
  liveInterval: null,

  async pollStatus() {
    try {
      const res = await fetch('/api/status');
      const d = await res.json();
      
      // If webcam is active but streamActive is false, auto-resume (e.g. on page refresh)
      if (d.webcam_active && !this.streamActive) {
        this.streamActive = true;
        this.resumeUI();
      }

      // Sync ROI zones if any
      if (d.roi_zones > 0 && this.roiZones.length === 0) {
          // Future: Fetch specific zones from an endpoint if needed
      }

    } catch(e) { console.error("Status Poll Error", e); }
  },

  async pollLiveDetections() {
    if (!this.streamActive) return;
    try {
      const res = await fetch('/api/live_detections');
      const dets = await res.json();
      this.renderLiveDetections(dets);
    } catch(e) {}
  },

  renderLiveDetections(dets) {
    const list = document.getElementById('detectionList');
    if (!list) return;

    const badge = document.getElementById('detectionCountBadge');
    if (badge) badge.textContent = `${dets.length} ACTIVE`;

    const totalMet = document.getElementById('metricTotal');
    if (totalMet) totalMet.textContent = dets.length;

    if (dets.length === 0) {
      const msg = this.streamActive ? 'SYSTEM_ARMED // SCANNING... NO_OBJECTS_DETECTED' : 'AWAITING_DATA_STREAM...';
      list.innerHTML = `<p class="text-center py-8 text-slate-600 font-mono text-[10px]">${msg}</p>`;
      return;
    }

    list.innerHTML = dets.map(d => {
      const riskColor = d.risk_level === 'High' ? 'emerald' : (d.risk_level === 'Medium' ? 'blue' : 'slate');
      const borderColor = d.risk_level === 'High' ? 'border-emerald-500/30' : 'border-white/5';
      
      return `
        <div class="bg-surface-container-high p-3 rounded-sm border ${borderColor} group cursor-pointer hover:bg-surface-variant transition-colors">
          <div class="flex justify-between items-start mb-2">
            <div>
              <h4 class="text-[11px] font-mono font-bold text-emerald-400 uppercase">${d.class_name}</h4>
              <p class="text-[9px] text-slate-500 font-mono">ID: ${d.detection_id.slice(-8)}</p>
            </div>
            <div class="px-2 py-0.5 bg-emerald-500/10 rounded-full border border-emerald-500/20">
              <span class="text-[8px] font-mono text-emerald-500 font-bold uppercase">${d.risk_level} Risk</span>
            </div>
          </div>
          <div class="flex justify-between items-center text-[9px] font-mono text-slate-400">
            <span>LIVE_BUFFER</span>
            <span class="text-emerald-500">${(d.confidence * 100).toFixed(1)}% CONF.</span>
          </div>
        </div>
      `;
    }).join('');
  },

  resumeUI() {
    const icon = document.getElementById('toggleIcon');
    const label = document.getElementById('streamStatusLabel');
    const img = document.getElementById('mainFeed');
    const placeholder = document.getElementById('feedPlaceholder');
    
    if (img) {
      img.src = '/stream?' + Date.now();
      img.classList.remove('hidden');
      if (placeholder) placeholder.classList.add('hidden');
      if (icon) icon.textContent = 'pause_circle';
      if (label) {
          label.textContent = 'REC [●] LIVE';
          label.classList.add('text-emerald-500');
          label.classList.remove('text-slate-500');
      }
      if (!this.liveInterval) {
        this.liveInterval = setInterval(() => this.pollLiveDetections(), 500);
      }
    }
  },

  async toggleStream() {
    const img = document.getElementById('mainFeed');
    const placeholder = document.getElementById('feedPlaceholder');
    const icon = document.getElementById('toggleIcon');
    const label = document.getElementById('streamStatusLabel');

    if (!this.streamActive) {
      try {
        await fetch('/stream/start', { method: 'POST' });
        this.streamActive = true;
        this.resumeUI();
      } catch(e) { console.error(e); }
    } else {
      try {
        await fetch('/stream/stop', { method: 'POST' });
        this.streamActive = false;
        if (img) {
            img.src = '';
            img.classList.add('hidden');
        }
        if (placeholder) placeholder.classList.remove('hidden');
        if (icon) icon.textContent = 'play_circle';
        if (label) {
            label.textContent = 'REC [OFF] STANDBY';
            label.classList.remove('text-emerald-500');
            label.classList.add('text-slate-500');
        }
        
        clearInterval(this.liveInterval);
        this.liveInterval = null;
        this.renderLiveDetections([]);
      } catch(e) { console.error(e); }
    }
  },

  roiDrawing: false,
  roiPoints: [],
  roiZones: [],

  setupROI() {
    const canvas = document.getElementById('roiCanvas');
    const outer = document.getElementById('streamOuter');
    if (!canvas || !outer) return;

    const resize = () => {
      canvas.width = outer.clientWidth;
      canvas.height = outer.clientHeight;
      this.drawROIZones();
    };
    resize();
    window.onresize = resize;

    canvas.onclick = (e) => {
      if (!this.roiDrawing) return;
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      const y = (e.clientY - rect.top) / rect.height;
      this.roiPoints.push([x, y]);
      this.drawROIZones();
    };

    canvas.oncontextmenu = (e) => {
      if (!this.roiDrawing) return;
      e.preventDefault();
      this.sealROIZone();
    };
  },

  drawROIZones() {
    const canvas = document.getElementById('roiCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Existing zones
    this.roiZones.forEach(z => {
      ctx.beginPath();
      ctx.strokeStyle = '#4edea3';
      ctx.fillStyle = 'rgba(78, 222, 163, 0.1)';
      z.forEach((p, i) => {
        const x = p[0] * canvas.width;
        const y = p[1] * canvas.height;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.stroke();
      ctx.fill();
    });

    // Current zone points
    if (this.roiPoints.length > 0) {
      ctx.beginPath();
      ctx.strokeStyle = '#fbbf24';
      this.roiPoints.forEach((p, i) => {
        const x = p[0] * canvas.width;
        const y = p[1] * canvas.height;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        ctx.arc(x, y, 3, 0, Math.PI * 2);
      });
      ctx.stroke();
    }
  },

  async sealROIZone() {
    if (this.roiPoints.length < 3) return;
    this.roiZones.push([...this.roiPoints]);
    this.roiPoints = [];
    this.drawROIZones();
    await fetch('/set_roi', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ zones: this.roiZones })
    });
    this.toggleROIDrawing(false);
  },

  toggleROIDrawing(force) {
    this.roiDrawing = force !== undefined ? force : !this.roiDrawing;
    const canvas = document.getElementById('roiCanvas');
    const hint = document.getElementById('roiHint');
    const btn = document.getElementById('toggleROI');
    
    if (this.roiDrawing) {
      canvas.classList.remove('hidden');
      hint.classList.remove('hidden');
      btn.textContent = 'SEAL_ZONE';
      btn.classList.add('text-yellow-400');
    } else {
      if (this.roiPoints.length > 0) this.sealROIZone();
      canvas.classList.add('hidden');
      hint.classList.add('hidden');
      btn.textContent = 'DRAW_ROI';
      btn.classList.remove('text-yellow-400');
    }
  },

  async clearROI() {
    this.roiZones = [];
    this.roiPoints = [];
    this.drawROIZones();
    await fetch('/clear_roi', { method: 'POST' });
    this.toggleROIDrawing(false);
  },

  init() {
    this.statusInterval = setInterval(() => this.pollStatus(), 3000);
    this.pollStatus();

    // Wire up Live page
    const toggleBtn = document.getElementById('toggleStream');
    if (toggleBtn) toggleBtn.onclick = () => this.toggleStream();

    const clearBtn = document.getElementById('clearROI');
    if (clearBtn) clearBtn.onclick = () => this.clearROI();

    const roiBtn = document.getElementById('toggleROI');
    if (roiBtn) roiBtn.onclick = () => this.toggleROIDrawing();

    this.setupROI();
  }
};

document.addEventListener('DOMContentLoaded', () => Sentinel.init());
