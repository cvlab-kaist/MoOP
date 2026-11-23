// Theme toggle (remembered per browser), active section in the nav, BibTeX copy, play videos when visible.
(function () {
  const root = document.documentElement;
  const dark = () => root.dataset.theme === "dark" ||
    (!root.dataset.theme && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const next = dark() ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("moop-theme", next); } catch (e) { /* storage unavailable */ }
  });

  const links = [...document.querySelectorAll(".topnav nav a")];
  const byId = Object.fromEntries(links.map((a) => [a.getAttribute("href").slice(1), a]));
  if ("IntersectionObserver" in window) {
    const seen = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting && byId[e.target.id]) {
          links.forEach((a) => a.classList.remove("active"));
          byId[e.target.id].classList.add("active");
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    document.querySelectorAll("main section[id]").forEach((s) => seen.observe(s));

    const play = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        const v = e.target;
        if (e.isIntersecting) {
          if (v.preload === "none") v.preload = "auto";
          v.play().catch(() => {});
        } else {
          v.pause();
        }
      });
    }, { threshold: 0.35 });
    document.querySelectorAll("video").forEach((v) => play.observe(v));
  }

  document.querySelectorAll(".copy").forEach((btn) => {
    const target = btn.dataset.copy || "bib-text";
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(document.getElementById(target).textContent);
        btn.textContent = "Copied";
      } catch (e) {
        btn.textContent = "Select and copy";
      }
      setTimeout(() => { btn.textContent = "Copy"; }, 1600);
    });
  });
})();
