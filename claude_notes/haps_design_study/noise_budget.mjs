// Noise-limited basal SNR for HAPS design candidates, using the mission design
// tool's own link budget (physics.js) so the numbers are comparable to it.
//   node claude_notes/haps_design_study/noise_budget.mjs designs.json
// Design: {name, f0_MHz, bw_MHz, T_us, n_el, taper, span_m?, window?}
// Fixed: 14 km, 20 m/s, 200 W payload @ 50% PA eff (tx power auto from duty
// cycle), 3 dB NF, 6.4 dB system loss, ice 1000 m (westcoast pilot), RSSNR mu
// = MU_DB (line property, common to all designs; only the deltas matter).
import { readFileSync } from 'node:fs';
import { scalars, basalSNR } from '/home/thomasteisberg/Documents/opr/radar_return_statistics_postprocessing/mission_design_tool/physics.js';

const MU_DB = Number(process.env.MU_DB ?? 60);
const ELEMENT_GAIN_DBI = 5;   // real element over a structure; common to all
const THICK_M = 1000;

function weights(kind, n) {
  if (kind === 'uniform' || !kind) return Array(n).fill(1);
  if (kind === 'hann') return Array.from({length: n}, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * (i + 1) / (n + 1)));
  if (kind === 'hamming') return Array.from({length: n}, (_, i) => 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (n - 1)));
  throw new Error(kind);
}
// Directivity of a line array of isotropic elements (exact sphere integral).
function directivity_dBi(w, dLam) {
  const n = w.length; let num = 0, den = 0;
  for (let m = 0; m < n; m++) for (let k = 0; k < n; k++) {
    const x = 2 * dLam * (m - k);
    den += w[m] * w[k] * (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x));
  }
  for (const v of w) num += v;
  return 10 * Math.log10(num * num / den);
}

const designs = JSON.parse(readFileSync(process.argv[2], 'utf8'));
console.log('design              f0   B    T   Gt    Gr   Ptx   TB    az_m  Nint  Pcomp  Tsys  bedSNR(mu=' + MU_DB + ')');
for (const d of designs) {
  const lam = 299792458 / (d.f0_MHz * 1e6);
  const span = d.span_m ?? 10, n = d.n_el;
  const dLam = n > 1 ? span / ((n - 1) * lam) : 0;
  const gt = ELEMENT_GAIN_DBI + directivity_dBi(weights(d.taper_tx ?? d.taper, n), dLam);
  const gr = ELEMENT_GAIN_DBI + directivity_dBi(weights(d.taper_rx ?? d.taper, n), dLam);
  const p = {
    altitude_m: 14000, velocity_ms: 20, payload_power_W: 200, pa_efficiency: 0.5,
    pulse_length_s: d.T_us * 1e-6, frequency_Hz: d.f0_MHz * 1e6, bandwidth_Hz: d.bw_MHz * 1e6,
    gain_tx_dBi: gt, gain_rx_dBi: gr, system_loss_dB: 6.4, noise_figure_dB: 3,
    surface_reflectivity_dB: -10, max_ice_thickness_m: 3400, epsilon_r: 3.15,
    radar_equation: 'infinite', overlap_mode: 'sidelobe', sidelobe_window: d.window ?? 'hann',
    tx_power_W: null, pri_s: null, noise_temp_K: null,
    azimuth_distance_m: d.az_m ?? null,
  };
  const s = scalars(p, { pulse_length_s: d.T_us * 1e-6, ...(d.az_m ? { azimuth_distance_m: d.az_m } : {}) });
  const out = new Float32Array(1);
  basalSNR(s, p, Uint16Array.from([THICK_M]), Float32Array.from([MU_DB]), out);
  console.log(`${d.name.padEnd(18)} ${String(d.f0_MHz).padStart(4)} ${String(d.bw_MHz).padStart(4)} ${String(d.T_us).padStart(3)} ` +
    `${gt.toFixed(1).padStart(5)} ${gr.toFixed(1).padStart(5)} ${String(Math.round(s.tx_power_W)).padStart(5)} ` +
    `${Math.round(p.bandwidth_Hz * s.pulse_length_s).toString().padStart(5)} ${s.azimuth_distance_m.toFixed(0).padStart(6)} ` +
    `${String(s.pulses_integrated).padStart(5)} ${s.pulse_compression_gain_dB.toFixed(1).padStart(6)} ${Math.round(s.system_temp_K).toString().padStart(5)}  ${out[0].toFixed(1)}`);
}
