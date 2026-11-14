let samplePresets = {};

document.addEventListener("DOMContentLoaded", async () => {
  if (window.lucide) {
    lucide.createIcons();
  }
  await fetchPresets();
  await fetchBenchmarkMetrics();
});

function switchTab(tabId) {
  const tabs = ['realtime', 'batch', 'benchmark'];
  tabs.forEach(t => {
    const el = document.getElementById(`tab-${t}`);
    const btn = document.getElementById(`tab-btn-${t}`);
    if (t === tabId) {
      el.classList.remove('hidden');
      btn.classList.add('border-cyan-500', 'text-cyan-400');
      btn.classList.remove('border-transparent', 'text-slate-400');
    } else {
      el.classList.add('hidden');
      btn.classList.remove('border-cyan-500', 'text-cyan-400');
      btn.classList.add('border-transparent', 'text-slate-400');
    }
  });
  if (window.lucide) lucide.createIcons();
}

async function fetchPresets() {
  try {
    const res = await fetch('/api/sample-presets');
    if (res.ok) {
      samplePresets = await res.json();
    }
  } catch (err) {
    console.warn("Could not fetch presets:", err);
  }
}

function loadPreset(type) {
  let data = null;
  // Realistic Indian Geolocation presets (Mumbai, Delhi, Bengaluru)
  if (type === 'legit') {
    data = {
      amt: 450.00,
      category: 'grocery_pos',
      gender: 'F',
      city_pop: 12500000,
      lat: 19.0760,     // Mumbai Cardholder
      long: 72.8777,
      merch_lat: 19.0820, // Local Mumbai Grocery POS (0.7 km away)
      merch_long: 72.8810,
      trans_date_trans_time: '2026-11-14 14:15:00',
      dob: '1992-04-10'
    };
  } else if (type === 'fraud') {
    data = {
      amt: 85000.00,
      category: 'shopping_net',
      gender: 'M',
      city_pop: 45000,
      lat: 12.9716,     // Bengaluru User
      long: 77.5946,
      merch_lat: 28.6139, // Cross-country New Delhi Merchant (1,750 km away at 3 AM!)
      merch_long: 77.2090,
      trans_date_trans_time: '2026-11-14 03:15:00',
      dob: '1975-08-22'
    };
  } else {
    data = {
      amt: 24999.00,
      category: 'misc_net',
      gender: 'M',
      city_pop: 2500000,
      lat: 28.6139,
      long: 77.2090,
      merch_lat: 28.6500,
      merch_long: 77.2400,
      trans_date_trans_time: '2026-11-14 22:40:00',
      dob: '1984-11-03'
    };
  }

  document.getElementById('input-amt').value = data.amt || 450.0;
  document.getElementById('input-category').value = data.category || 'shopping_net';
  document.getElementById('input-datetime').value = data.trans_date_trans_time || '2026-11-14 14:30:00';
  document.getElementById('input-dob').value = data.dob || '1992-01-01';
  document.getElementById('input-gender').value = (data.gender || 'F').toUpperCase() === 'M' ? 'M' : 'F';
  document.getElementById('input-city-pop').value = data.city_pop || 1200000;
  document.getElementById('input-lat').value = data.lat;
  document.getElementById('input-long').value = data.long;
  document.getElementById('input-merch-lat').value = data.merch_lat;
  document.getElementById('input-merch-long').value = data.merch_long;

  // Auto trigger analysis
  document.getElementById('tx-form').dispatchEvent(new Event('submit'));
}

async function handleSinglePrediction(e) {
  e.preventDefault();
  const btn = document.getElementById('btn-analyze');
  const originalText = btn.innerHTML;
  btn.innerHTML = `<span>ANALYZING INFERENCE...</span>`;
  btn.disabled = true;

  const payload = {
    amt: parseFloat(document.getElementById('input-amt').value),
    category: document.getElementById('input-category').value,
    gender: document.getElementById('input-gender').value,
    city_pop: parseFloat(document.getElementById('input-city-pop').value),
    lat: parseFloat(document.getElementById('input-lat').value),
    long: parseFloat(document.getElementById('input-long').value),
    merch_lat: parseFloat(document.getElementById('input-merch-lat').value),
    merch_long: parseFloat(document.getElementById('input-merch-long').value),
    trans_date_trans_time: document.getElementById('input-datetime').value,
    dob: document.getElementById('input-dob').value
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error("Inference failed");
    const data = await res.json();
    updateVerdictUI(data);
  } catch (err) {
    alert("Inference Error: " + err.message);
  } finally {
    btn.innerHTML = originalText;
    btn.disabled = false;
    if (window.lucide) lucide.createIcons();
  }
}

function updateVerdictUI(data) {
  const pct = data.risk_score_pct;
  const scoreEl = document.getElementById('risk-score-pct');
  const barEl = document.getElementById('risk-bar');
  const badgeEl = document.getElementById('risk-score-badge');
  const actionEl = document.getElementById('res-action');
  const latencyEl = document.getElementById('res-latency');

  scoreEl.textContent = pct.toFixed(1);
  barEl.style.width = `${Math.max(pct, 5)}%`;
  latencyEl.textContent = `Inference: ${data.latency_ms} ms`;
  actionEl.textContent = data.action_recommendation;

  if (data.features_computed) {
    document.getElementById('stat-dist').textContent = `${data.features_computed.distance_km} km`;
    document.getElementById('stat-window').textContent = data.features_computed.is_night ? 'Night (High-Risk)' : 'Day';
    document.getElementById('stat-age').textContent = `${data.features_computed.age} yrs`;
  }

  if (data.risk_level === "HIGH") {
    scoreEl.className = "text-6xl font-black tracking-tight text-rose-500 font-mono";
    barEl.className = "bg-rose-500 h-2 rounded-full transition-all duration-700";
    badgeEl.className = "inline-flex items-center space-x-2 px-4 py-1.5 rounded-full text-xs font-mono font-bold bg-rose-950/80 text-rose-300 border border-rose-700 animate-pulse";
    badgeEl.innerHTML = `<span class="w-2 h-2 rounded-full bg-rose-500"></span><span>🚨 HIGH RISK FRAUD DETECTED</span>`;
  } else if (data.risk_level === "MEDIUM") {
    scoreEl.className = "text-6xl font-black tracking-tight text-amber-400 font-mono";
    barEl.className = "bg-amber-400 h-2 rounded-full transition-all duration-700";
    badgeEl.className = "inline-flex items-center space-x-2 px-4 py-1.5 rounded-full text-xs font-mono font-bold bg-amber-950/80 text-amber-300 border border-amber-700";
    badgeEl.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400"></span><span>⚠️ SUSPICIOUS / OTP CHALLENGE</span>`;
  } else {
    scoreEl.className = "text-6xl font-black tracking-tight text-emerald-400 font-mono";
    barEl.className = "bg-emerald-400 h-2 rounded-full transition-all duration-700";
    badgeEl.className = "inline-flex items-center space-x-2 px-4 py-1.5 rounded-full text-xs font-mono font-bold bg-emerald-950/80 text-emerald-300 border border-emerald-700";
    badgeEl.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400"></span><span>✓ APPROVED / LOW RISK</span>`;
  }
}

async function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  const section = document.getElementById('batch-results-section');
  section.classList.remove('hidden');

  try {
    const res = await fetch('/api/batch-predict', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error("Batch processing failed");
    const data = await res.json();

    document.getElementById('b-total').textContent = data.total_processed.toLocaleString();
    document.getElementById('b-approved').textContent = data.approved_count.toLocaleString();
    document.getElementById('b-flagged').textContent = data.flagged_fraud_count.toLocaleString();
    document.getElementById('b-rate').textContent = `${data.fraud_rate_pct}%`;

    const tbody = document.getElementById('batch-table-body');
    tbody.innerHTML = '';

    if (data.sample_flagged_rows && data.sample_flagged_rows.length > 0) {
      data.sample_flagged_rows.forEach(r => {
        const tr = document.createElement('tr');
        tr.className = "bg-rose-950/20 hover:bg-rose-950/30 transition";
        tr.innerHTML = `
          <td class="px-4 py-2 font-mono font-bold text-white">₹${r.amt.toFixed(2)}</td>
          <td class="px-4 py-2 text-slate-300">${r.category}</td>
          <td class="px-4 py-2 font-mono text-rose-400 font-bold">${r.risk_score_pct}%</td>
          <td class="px-4 py-2 font-mono"><span class="px-2 py-0.5 rounded bg-rose-900/60 text-rose-300 text-[10px] font-bold">FLAGGED</span></td>
        `;
        tbody.appendChild(tr);
      });
    } else {
      tbody.innerHTML = `<tr><td colspan="4" class="px-4 py-4 text-center text-slate-500">No fraudulent transactions detected in this batch.</td></tr>`;
    }

  } catch (err) {
    alert("Batch Upload Error: " + err.message);
  }
}

async function fetchBenchmarkMetrics() {
  try {
    const res = await fetch('/api/metrics');
    if (!res.ok) return;
    const data = await res.json();
    const tbody = document.getElementById('benchmark-tbody');
    tbody.innerHTML = '';

    data.results.forEach((r, idx) => {
      const isChamp = idx === 3;
      const tr = document.createElement('tr');
      tr.className = isChamp ? "bg-emerald-950/20 border-l-4 border-emerald-500 font-semibold" : "hover:bg-slate-900/40 transition";
      tr.innerHTML = `
        <td class="px-4 py-3 text-white flex items-center space-x-2">
          ${isChamp ? '<span class="text-xs text-emerald-400 font-bold">🏆</span>' : ''}
          <span>${r.model_name}</span>
        </td>
        <td class="px-4 py-3 font-mono text-slate-300">${(r.accuracy * 100).toFixed(2)}%</td>
        <td class="px-4 py-3 font-mono text-slate-300">${(r.precision * 100).toFixed(2)}%</td>
        <td class="px-4 py-3 font-mono ${isChamp ? 'text-emerald-400 font-bold' : 'text-slate-300'}">${(r.recall * 100).toFixed(2)}%</td>
        <td class="px-4 py-3 font-mono text-slate-300">${r.f1_score.toFixed(3)}</td>
        <td class="px-4 py-3 font-mono text-slate-300">${r.roc_auc.toFixed(3)}</td>
        <td class="px-4 py-3 font-mono ${isChamp ? 'text-emerald-400 font-bold' : 'text-slate-300'}">${r.pr_auc.toFixed(3)}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.warn("Could not fetch metrics:", err);
  }
}