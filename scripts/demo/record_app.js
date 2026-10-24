// Screen-record the gallery app making an effect, for the demo video.
//   node scripts/demo/record_app.js out_dir [http://127.0.0.1:7860]
// Writes app_flow.webm plus marks.json (seconds at which each step happened).
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const OUT = process.argv[2] || 'rec';
const URL = process.argv[3] || 'http://127.0.0.1:7860';
const W = 1280, H = 720;  // CSS px; recorded at 1.5x = 1920x1080

// a visible cursor + click ripple, since headless recordings have no pointer
const CURSOR = `
  (() => {
    const c = document.createElement('div');
    c.style.cssText = 'position:fixed;z-index:99999;width:22px;height:22px;margin:-4px 0 0 -4px;pointer-events:none;transition:transform .08s;' +
      'background:url("data:image/svg+xml,' + encodeURIComponent('<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'22\\' height=\\'22\\' viewBox=\\'0 0 22 22\\'><path d=\\'M3 2l14 9-6 1.5L8 19z\\' fill=\\'white\\' stroke=\\'black\\' stroke-width=\\'1.4\\'/></svg>') + '") no-repeat;';
    document.addEventListener('DOMContentLoaded', () => document.body.appendChild(c));
    window.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
    window.addEventListener('mousedown', e => {
      c.style.transform = 'scale(.85)';
      const r = document.createElement('div');
      r.style.cssText = 'position:fixed;z-index:99998;pointer-events:none;border-radius:50%;border:2px solid #4de1ff;width:10px;height:10px;' +
        'left:' + (e.clientX - 5) + 'px;top:' + (e.clientY - 5) + 'px;transition:all .45s ease-out;opacity:1';
      document.body.appendChild(r);
      requestAnimationFrame(() => { r.style.width = r.style.height = '46px'; r.style.left = (e.clientX - 23) + 'px'; r.style.top = (e.clientY - 23) + 'px'; r.style.opacity = '0'; });
      setTimeout(() => r.remove(), 600);
    }, true);
    window.addEventListener('mouseup', () => { c.style.transform = ''; }, true);
  })();`;

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const context = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 1.5, recordVideo: { dir: OUT, size: { width: 1920, height: 1080 } } });
  await context.addInitScript(CURSOR);
  const page = await context.newPage();
  const t0 = Date.now();
  const marks = {};
  const mark = (k) => { marks[k] = (Date.now() - t0) / 1000; console.log(k, marks[k]); };
  const glide = async (sel, steps = 25) => {
    await page.locator(sel).first().scrollIntoViewIfNeeded();
    await page.waitForTimeout(250);
    const b = await page.locator(sel).first().boundingBox();
    await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2, { steps });
    return b;
  };

  await page.goto(URL, { waitUntil: 'networkidle' });
  await page.mouse.move(W / 2, H / 2);
  mark('home');
  await page.waitForTimeout(2500);
  for (let i = 0; i < 4; i++) { await glide(`.card >> nth=${i}`, 18); await page.waitForTimeout(700); }
  await glide('.card >> nth=0');
  mark('open_lidar');
  await page.locator('.card >> nth=0').click();
  await page.waitForTimeout(1200);
  await glide('.sample:has-text("rock field")');
  await page.locator('.sample:has-text("rock field")').click();
  mark('picked_clip');
  await page.waitForTimeout(900);
  await glide('.chip:has-text("gold")');
  await page.locator('.chip:has-text("gold")').click();
  await page.waitForTimeout(600);
  await glide('.chip:has-text("crane up")');
  await page.locator('.chip:has-text("crane up")').click();
  await page.waitForTimeout(600);
  await page.locator('#duration').fill('5');
  await page.locator('#duration').dispatchEvent('input');
  await page.waitForTimeout(500);
  await glide('#go');
  await page.waitForTimeout(400);
  mark('generate');
  await page.locator('#go').click();
  await page.waitForSelector('#result:not([hidden]), #error:not([hidden])', { timeout: 900000 });
  mark('result');
  const err = await page.locator('#error').textContent();
  if (err && err.trim()) console.log('ERROR', err);
  await page.mouse.move(W - 400, H - 200, { steps: 20 });
  await page.waitForTimeout(9000);
  await glide('#wf');
  await page.waitForTimeout(1500);
  mark('end');
  await context.close();
  await browser.close();
  const vid = fs.readdirSync(OUT).filter(f => f.endsWith('.webm')).map(f => path.join(OUT, f)).sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs)[0];
  fs.renameSync(vid, path.join(OUT, 'app_flow.webm'));
  fs.writeFileSync(path.join(OUT, 'marks.json'), JSON.stringify(marks, null, 1));
})().catch(e => { console.error(e); process.exit(1); });
