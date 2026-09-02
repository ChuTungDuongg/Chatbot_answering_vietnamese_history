import { test, expect } from "@playwright/test";

test("blossom remains clear at 16–40px in dark and light themes", async ({ page }, info) => {
  await page.route("**/api/v1/**", (route) => route.fulfill({ json: [] }));
  await page.goto("/");
  const welcome = page.locator(".welcome-identity .brand-mark");
  await expect(welcome).toBeVisible();
  await expect(welcome).toHaveAttribute("aria-hidden", "true");
  await expect(page.locator(".logo-mark")).toHaveCount(0);
  // A visual specimen made from the rendered production component, not a second icon.
  await page.evaluate(() => {
    const specimen = document.createElement("div");
    specimen.id = "brand-specimen";
    specimen.style.cssText = "position:fixed;inset:0;z-index:1000;background:var(--bg);color:var(--text-primary);display:flex;flex-wrap:wrap;align-content:center;justify-content:center;gap:24px;padding:32px";
    for (const size of [16, 20, 24, 32, 40]) {
      const cell = document.createElement("div");
      cell.style.cssText = "display:grid;justify-items:center;align-content:center;gap:20px;width:72px;height:104px";
      const icon = document.querySelector(".welcome-identity .brand-mark").cloneNode(true);
      icon.setAttribute("width", size);
      icon.setAttribute("height", size);
      cell.append(icon, document.createTextNode(`${size} px`));
      specimen.append(cell);
    }
    document.body.append(specimen);
  });
  for (const theme of ["dark", "light"]) {
    await page.evaluate((value) => { document.documentElement.dataset.theme = value; }, theme);
    const metrics = await page.locator("#brand-specimen .brand-mark").evaluateAll((icons) => {
      const luminance = (rgb) => {
        const channels = rgb.match(/[\d.]+/g).slice(0, 3).map((v) => Number(v) / 255);
        const linear = channels.map((v) => v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4);
        return linear[0] * .2126 + linear[1] * .7152 + linear[2] * .0722;
      };
      const contrast = (a, b) => (Math.max(a, b) + .05) / (Math.min(a, b) + .05);
      const background = luminance(getComputedStyle(document.getElementById("brand-specimen")).backgroundColor);
      return icons.map((icon) => {
        const bbox = icon.getBBox();
        const rect = icon.getBoundingClientRect();
        const petals = icon.querySelector(".brand-mark-petals");
        const center = icon.querySelector("circle");
        return { size: Number(icon.getAttribute("width")), width: rect.width, height: rect.height,
          clipped: bbox.x < 0 || bbox.y < 0 || bbox.x + bbox.width > 64 || bbox.y + bbox.height > 64,
          petals: petals.children.length, centerRadius: center.r.baseVal.value,
          petalContrast: contrast(luminance(getComputedStyle(petals).fill), background),
          centerContrast: contrast(luminance(getComputedStyle(center).fill), background) };
      });
    });
    for (const mark of metrics) {
      expect(mark.width).toBe(mark.size);
      expect(mark.height).toBe(mark.size);
      expect(mark.clipped).toBe(false);
      expect(mark.petals).toBe(5);
      expect(mark.centerRadius).toBeGreaterThan(0);
      expect(mark.petalContrast).toBeGreaterThan(3);
      expect(mark.centerContrast).toBeGreaterThan(3);
    }
    await page.locator("#brand-specimen").screenshot({ path: info.outputPath(`brand-${theme}.png`) });
  }
});
