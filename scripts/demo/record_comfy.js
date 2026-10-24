// Screen-record a tour of the LiDAR Scan graph and its App Mode view in ComfyUI.
//   node scripts/demo/record_comfy.js out_dir [http://127.0.0.1:8188]
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const OUT = process.argv[2] || 'rec';
const URL = process.argv[3] || 'http://127.0.0.1:8188';
const ROOT = path.resolve(__dirname, '..', '..');
const W = 1920, H = 1080;

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const context = await browser.newContext({ viewport: { width: W, height: H }, recordVideo: { dir: OUT, size: { width: W, height: H } } });
  const page = await context.newPage();
  const t0 = Date.now();
  const marks = {};
  const mark = (k) => { marks[k] = (Date.now() - t0) / 1000; console.log(k, marks[k]); };
  await page.goto(URL, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
  await page.waitForTimeout(2000);
  const graph = JSON.parse(fs.readFileSync(path.join(ROOT, 'workflows', 'lidar_scan.json'), 'utf8'));
  await page.evaluate(async (wf) => { await window.app.loadGraphData(wf); }, graph);
  await page.waitForTimeout(1200);
  // close any toasts / side panels that cover the canvas
  await page.keyboard.press('Escape');
  // fly along the pipeline left -> right
  const span = await page.evaluate(() => {
    const ns = window.app.graph._nodes;
    const xs = ns.map(n => n.pos[0]), ys = ns.map(n => n.pos[1]);
    return { x0: Math.min(...xs), x1: Math.max(...ns.map(n => n.pos[0] + n.size[0])), y0: Math.min(...ys),
             y1: Math.max(...ns.map(n => n.pos[1] + n.size[1])) };
  });
  mark('graph');
  await page.evaluate(async ({ span, W }) => {
    const c = window.app.canvas;
    const scale = 1.5;
    c.ds.scale = scale;
    const startX = -span.x0 + 50, endX = -(span.x1 - W / scale) - 50;
    const y = (1080 / scale - (span.y1 - span.y0)) / 2 - span.y0 + 10;
    const dur = 16000, t0 = performance.now();
    await new Promise(res => {
      const step = (t) => {
        const k = Math.min(1, (t - t0) / dur);
        const e = 0.5 - 0.5 * Math.cos(Math.PI * k);
        c.ds.offset = [startX + (endX - startX) * e, y];
        c.setDirty(true, true);
        if (k < 1) requestAnimationFrame(step); else res();
      };
      requestAnimationFrame(step);
    });
  }, { span, W });
  await page.waitForTimeout(800);
  mark('graph_end');
  const appwf = JSON.parse(fs.readFileSync(path.join(ROOT, 'workflows', 'lidar_scan.app.json'), 'utf8'));
  await page.evaluate(async (wf) => { await window.app.loadGraphData(wf); }, appwf);
  await page.waitForTimeout(1500);
  const skip = page.locator('button:has-text("Skip")');
  if (await skip.count()) await skip.first().click();
  mark('appmode');
  await page.waitForTimeout(6000);
  mark('end');
  await context.close();
  await browser.close();
  const vid = fs.readdirSync(OUT).filter(f => f.endsWith('.webm') && !f.startsWith('app_flow')).map(f => path.join(OUT, f))
    .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs)[0];
  fs.renameSync(vid, path.join(OUT, 'comfy_tour.webm'));
  fs.writeFileSync(path.join(OUT, 'comfy_marks.json'), JSON.stringify(marks, null, 1));
})().catch(e => { console.error(e); process.exit(1); });
