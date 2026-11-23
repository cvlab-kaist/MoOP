// Latency breakdown, LingoQA examples and closed-loop driving replays. Plain SVG/DOM, rendered at the container's pixel width.
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {}, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  };

  // one tooltip for every chart; marks carry their text in data-tip
  const tip = document.createElement("div");
  tip.className = "viz-tip";
  document.body.appendChild(tip);
  const showTip = (ev) => {
    const t = ev.target.closest("[data-tip]");
    if (!t) { tip.classList.remove("on"); return; }
    tip.innerHTML = t.getAttribute("data-tip");
    const x = Math.min(ev.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    const y = ev.clientY + 16 + tip.offsetHeight > window.innerHeight ? ev.clientY - tip.offsetHeight - 10 : ev.clientY + 16;
    tip.style.left = `${x}px`; tip.style.top = `${y}px`;
    tip.classList.add("on");
  };
  document.addEventListener("pointermove", showTip);
  document.addEventListener("pointerdown", showTip);
  document.addEventListener("scroll", () => tip.classList.remove("on"), { passive: true });

  const onResize = (fn) => {
    let w = 0;
    const run = () => { const nw = Math.round(document.documentElement.clientWidth); if (nw !== w) { w = nw; fn(); } };
    window.addEventListener("resize", run);
    run();
  };

  // ---------------- latency breakdown (paper Table 5) ----------------
  const LAT = [
    { m: "Dense", p: 691.3, d: 438.2, n: 212.3 },
    { m: "MoOP (VLM)", p: 515.6, d: 312.5, n: 204.3 },
    { m: "MoOP + expert 75%", p: 514.4, d: 312.3, n: 160.3 },
    { m: "MoOP + expert 87.5%", p: 513.4, d: 312.4, n: 153.8 },
    { m: "MoOP + expert 93.75%", p: 514.1, d: 312.8, n: 150.9 },
  ];
  function latency() {
    const svg = document.getElementById("lat-chart");
    const draw = () => {
      svg.innerHTML = "";
      const w = svg.parentElement.clientWidth - 18;
      const narrow = w < 560;
      const left = narrow ? 8 : 150, right = narrow ? 8 : 112, top = 24, row = 30, gap = 2;
      const H = top + LAT.length * (narrow ? row + 18 : row) + 8;
      svg.setAttribute("width", w); svg.setAttribute("height", H); svg.setAttribute("viewBox", `0 0 ${w} ${H}`);
      const span = w - left - right, max = 1400;
      [0, 250, 500, 750, 1000, 1250].forEach((t) => {
        const x = left + (t / max) * span;
        el("line", { x1: x, x2: x, y1: top - 6, y2: H - 4, stroke: "var(--grid)", "stroke-width": 1 }, svg);
        el("text", { x, y: top - 10, "text-anchor": "middle" }, svg).textContent = t === 0 ? "0 ms" : `${t}`;
      });
      LAT.forEach((r, i) => {
        const total = r.p + r.d + r.n;
        const y = top + i * (narrow ? row + 18 : row) + (narrow ? 16 : 4);
        const lab = el("text", { x: narrow ? left : left - 10, y: narrow ? y - 4 : y + 14, "text-anchor": narrow ? "start" : "end",
          class: i === 0 ? "" : "strong" }, svg);
        lab.textContent = r.m;
        let x = left;
        [["ce", r.p, "VLM prefill"], ["both", r.d, "CoC decoding (16 tokens)"], ["fm", r.n, "expert denoising (10 steps)"]].forEach(([k, v, name], j) => {
          const wseg = (v / max) * span;
          el("rect", { x, y, width: Math.max(wseg - (j < 2 ? gap : 0), 1), height: row - 10, rx: j === 2 ? 4 : 1,
            fill: `var(--s-${k})`, "data-tip": `<b>${r.m}</b><br>${name}: ${v.toFixed(1)} ms<br>end to end: ${total.toFixed(1)} ms` }, svg);
          x += wseg;
        });
        const sp = (1341.8 / total).toFixed(2);
        const t = el("text", { x: x + 6, y: y + 14, class: "strong" }, svg);
        t.textContent = narrow ? `${total.toFixed(0)} ms` : `${total.toFixed(1)} ms · ${sp}×`;
      });
    };
    onResize(draw);
  }

  // ---------------- LingoQA examples (selected) ----------------
  function lingo(data) {
    let i = 0;
    const list = document.getElementById("lq-list");
    document.getElementById("lq-acc").textContent =
      Object.entries(data.acc).sort((x, y) => y[1] - x[1]).map(([m, v]) => `${m} ${v.toFixed(1)}`).join(", ");
    const show = () => {
      const it = data.items[i];
      document.getElementById("lq-title").textContent = `Question ${i + 1} of ${data.items.length}`;
      const img = document.getElementById("lq-img"); img.src = it.img; img.alt = `Camera frame for: ${it.q}`;
      document.getElementById("lq-q").textContent = it.q;
      document.getElementById("lq-refs").textContent = it.refs.join("  |  ");
      list.innerHTML = "";
      it.a.forEach((a) => {
        const k = a.t.search(/<i\d/);                    // token ids leaking into the text: a degenerate continuation
        const text = (k < 0 ? a.t : a.t.slice(0, k)).trim();
        const row = document.createElement("div");
        row.className = "coc-row" + (a.m === "Dense" ? " ref" : "") + (a.m === "MoOP" ? " ours" : "") + (k >= 0 ? " degen" : "");
        const who = document.createElement("div"); who.className = "who"; who.textContent = a.m === "Dense" ? "Dense (reference)" : a.m;
        const txt = document.createElement("div"); txt.className = "txt"; txt.textContent = text;
        if (k >= 0) { const f = document.createElement("span"); f.className = "flag"; f.textContent = text ? "degenerate tokens" : "only degenerate tokens"; txt.appendChild(f); }
        const v = document.createElement("div"); v.className = "ade";
        const tag = document.createElement("span"); tag.className = `tag ${a.ok ? "pass" : "fail"}`; tag.textContent = a.ok ? "correct" : "wrong";
        v.appendChild(tag); v.title = "Lingo-Judge verdict";
        row.append(who, txt, v);
        list.appendChild(row);
      });
    };
    document.getElementById("lq-prev").addEventListener("click", () => { i = (i - 1 + data.items.length) % data.items.length; show(); });
    document.getElementById("lq-next").addEventListener("click", () => { i = (i + 1) % data.items.length; show(); });
    show();
  }

  // ---------------- closed-loop driving: the four driver cameras over the BEV ----------------
  const VERDICT_TXT = { pass: "no incident", collision: "at-fault collision", offroad: "off-road" };
  const KIND_TXT = { collision: "At-fault collision", offroad: "Off-road" };
  const BUDGET_TXT = { u40: "24.0% removed", u50: "30.0% removed" };
  function camera(data) {
    const video = document.getElementById("cam-video");
    const caps = document.getElementById("cam-caps");
    const tabs = document.getElementById("cam-tabs");
    let sc = null, boxes = [], last = -1;
    const cocAt = (runs, f) => { let t = runs[0][1]; for (const [s, x] of runs) { if (s > f) break; t = x; } return t; };
    const clean = (t) => { const k = t.indexOf("<|"); return k < 0 ? t : t.slice(0, k).trim(); };
    const setCoc = (el, t, wait) => {                // text after a special token is a degenerate continuation: flag it
      el.textContent = clean(t) || wait || "(empty reasoning output)";
      if (t.includes("<|")) { const s = document.createElement("span"); s.className = "flag"; s.textContent = "degenerate continuation"; el.appendChild(s); }
    };
    const stacked = window.matchMedia("(max-width: 640px)");
    const fit = () => boxes.forEach((b, j) => {      // side by side: reserve the longest text's height so the row never jumps
      if (stacked.matches) { b.coc.style.minHeight = "0"; return; }
      const longest = sc.arms[j].coc.reduce((a, r) => (clean(r[1]).length > clean(a).length ? r[1] : a), "");
      const cur = [...b.coc.childNodes]; b.coc.style.minHeight = "0"; setCoc(b.coc, longest);
      b.coc.style.minHeight = `${b.coc.offsetHeight}px`; b.coc.replaceChildren(...cur);
    });
    const update = () => {
      const f = Math.min(Math.floor(video.currentTime * data.fps + 1e-3), sc.n_frames - 1);
      if (f === last) return;
      last = f;
      document.getElementById("cam-time").textContent = `t = ${(sc.t_offset + f / data.fps).toFixed(1)} s`;
      sc.arms.forEach((a, j) => {
        const g = Math.min(f, a.n_frames - 1);
        const first = a.coc.find((r) => r[1]);        // before the model's first reasoning there is none yet
        setCoc(boxes[j].coc, cocAt(a.coc, g),
          first && g < first[0] ? `(no reasoning yet: the first arrives at ${(sc.t_offset + first[0] / data.fps).toFixed(1)} s)` : "");
        boxes[j].root.classList.toggle("hit", a.event_frame !== null && g >= a.event_frame);
      });
    };
    const loop = () => { update(); if (!video.paused) requestAnimationFrame(loop); };
    const show = (k) => {
      sc = data.scenes[k]; last = -1;
      tabs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.k === k)));
      document.getElementById("cam-title").textContent = KIND_TXT[sc.kind];
      document.getElementById("cam-sub").textContent =
        `· ${sc.budget} (${BUDGET_TXT[sc.budget]}) · ${sc.set} · scene ${sc.short} · ${sc.route_m} m route`;
      const playing = !video.paused;
      video.poster = `static/videos/${sc.poster}`; video.src = `static/videos/${sc.video}`;
      caps.style.gridTemplateColumns = `repeat(${sc.arms.length}, minmax(0, 1fr))`;
      caps.innerHTML = "";
      boxes = sc.arms.map((a) => {
        const root = document.createElement("div");
        root.className = "cam-cap" + (a.name === "MoOP" ? " ours" : "");
        const who = document.createElement("div"); who.className = "cam-who";
        const b = document.createElement("b"); b.textContent = a.name;
        const tag = document.createElement("span"); tag.className = `tag ${a.verdict === "pass" ? "pass" : "fail"}`;
        tag.textContent = a.verdict === "pass" ? `${VERDICT_TXT.pass} · ${a.dist_m} m` : `${VERDICT_TXT[a.verdict]} at ${a.event_t.toFixed(1)} s`;
        who.append(b, tag);
        const coc = document.createElement("div"); coc.className = "cam-coc";
        root.append(who, coc); caps.appendChild(root);
        return { root, coc };
      });
      fit(); update();
      if (playing) video.play().catch(() => {});
    };
    ["collision", "offroad"].forEach((kind) => {     // one row of tabs per failure type
      const row = document.createElement("div"); row.className = "tab-row";
      const lab = document.createElement("span"); lab.className = "tab-kind"; lab.textContent = KIND_TXT[kind];
      const btns = document.createElement("div"); btns.className = "tab-btns";
      row.append(lab, btns);
      let n = 0;
      data.scenes.forEach((s, k) => {
        if (s.kind !== kind) return;
        const b = document.createElement("button"); b.type = "button"; b.dataset.k = k;
        b.textContent = `${++n} · ${s.set} · ${s.budget}`;
        b.addEventListener("click", () => show(k));
        btns.appendChild(b);
      });
      if (n) tabs.appendChild(row);
    });
    ["play", "seeked", "timeupdate"].forEach((e) => video.addEventListener(e, () => (e === "play" ? loop() : update())));
    onResize(() => sc && fit());
    show(0);
    new IntersectionObserver((es, ob) => {           // start once, muted, when the block first scrolls into view
      if (es[0].isIntersecting) { video.play().catch(() => {}); ob.disconnect(); }
    }, { threshold: 0.5 }).observe(video);
  }

  latency();
  fetch("static/data/driving.json").then((r) => r.json()).then(camera).catch(() => {
    document.getElementById("cam-caps").textContent = "The driving replays could not be loaded.";
  });
  fetch("static/data/lingoqa.json").then((r) => r.json()).then(lingo).catch(() => {
    document.getElementById("lq-list").textContent = "The examples could not be loaded.";
  });
})();
