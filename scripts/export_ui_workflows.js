// Convert workflows/api/*.json into ComfyUI UI workflows (drag-and-drop .json)
// and App Mode workflows (.app.json) using the real ComfyUI frontend.
//
//   node scripts/export_ui_workflows.js [http://127.0.0.1:8188]
//
// Needs a running ComfyUI with LiDAR Lab installed and Playwright with Chromium.
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const SERVER = process.argv[2] || 'http://127.0.0.1:8188';
const ROOT = path.resolve(__dirname, '..');
const API_DIR = path.join(ROOT, 'workflows', 'api');
const OUT_DIR = path.join(ROOT, 'workflows');

// Widgets surfaced in App Mode for each effect node, in display order.
const APP_INPUTS = {
  LidarLabScan: ['style', 'camera', 'sweep', 'density', 'scan_start', 'hold'],
  LidarLabClones: ['mode', 'look', 'count', 'interval'],
  LidarLabVanish: ['particles', 'direction', 'start', 'duration'],
  LidarLabBulletTime: ['angle', 'duration', 'zoom', 'freeze_at'],
};

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1800, height: 1100 } });
  await page.goto(SERVER, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
  await page.waitForTimeout(2000);
  for (const file of fs.readdirSync(API_DIR).filter(f => f.endsWith('.json')).sort()) {
    const name = file.replace(/\.json$/, '');
    const api = JSON.parse(fs.readFileSync(path.join(API_DIR, file), 'utf8'));
    const ui = await page.evaluate(async ({ api, name }) => {
      await window.app.loadApiJson(api, `${name}.json`);
      await new Promise(r => setTimeout(r, 300));
      window.app.graph.arrange(60);
      return window.app.graph.serialize();
    }, { api, name });
    fs.writeFileSync(path.join(OUT_DIR, `${name}.json`), JSON.stringify(ui, null, 1));

    // App Mode: expose the clip, the trim and the effect's main knobs; show the saved video
    const byType = t => ui.nodes.filter(n => n.type === t).map(n => n.id);
    const inputs = [];
    for (const id of byType('LoadVideo')) inputs.push([id, 'file']);
    for (const id of byType('Video Slice')) inputs.push([id, 'start_time'], [id, 'duration']);
    for (const [type, widgets] of Object.entries(APP_INPUTS)) {
      for (const id of byType(type)) for (const w of widgets) inputs.push([id, w]);
    }
    const app = JSON.parse(JSON.stringify(ui));
    app.extra = app.extra || {};
    app.extra.linearMode = true;
    app.extra.linearData = { inputs, outputs: byType('SaveVideo') };
    fs.writeFileSync(path.join(OUT_DIR, `${name}.app.json`), JSON.stringify(app, null, 1));
    console.log(`wrote workflows/${name}.json and workflows/${name}.app.json (${ui.nodes.length} nodes)`);
  }
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
