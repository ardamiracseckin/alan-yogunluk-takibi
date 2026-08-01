/**
 * Dashboard istemcisi.
 *
 * Dışarıdan hiçbir kütüphane çekilmez: grafik doğrudan canvas'a çizilir.
 * Böylece sistem internetsiz bir makinede ve konteyner içinde de aynı
 * şekilde çalışır.
 */

const CANLI_ARALIK_MS = 1000;
const GECMIS_ARALIK_MS = 10000;
const ISI_ARALIK_MS = 15000;
const GECMIS_DAKIKA = 30;

const el = (id) => document.getElementById(id);

const sayiBicimi = new Intl.NumberFormat("tr-TR");
const saatBicimi = new Intl.DateTimeFormat("tr-TR", { hour: "2-digit", minute: "2-digit" });

/* --- canlı ölçümler --------------------------------------------------- */

async function canliGuncelle() {
  try {
    const veri = await (await fetch("/api/live")).json();

    el("doluluk").textContent = sayiBicimi.format(veri.total_people);
    el("giris").textContent = sayiBicimi.format(veri.total_in);
    el("cikis").textContent = sayiBicimi.format(veri.total_out);
    el("fps").textContent = `${veri.fps.toFixed(1)} FPS`;

    bolgeleriYaz(veri.counts, veri.density);
    durumuYaz(veri.source_ok);
  } catch (hata) {
    durumuYaz(false);
  }
}

function bolgeleriYaz(sayilar, yogunluklar) {
  const liste = el("bolgeler");
  const adlar = Object.keys(sayilar || {});

  if (adlar.length === 0) {
    liste.innerHTML = '<li class="bos">Bölge verisi bekleniyor…</li>';
    return;
  }

  liste.innerHTML = adlar
    .map((ad) => {
      const yogunluk = (yogunluklar && yogunluklar[ad]) || 0;
      return `<li><span>${escapeHtml(ad)}</span><span class="sayi">${sayilar[ad]} kişi · ${yogunluk.toFixed(
        2
      )}</span></li>`;
    })
    .join("");
}

function durumuYaz(kaynakTamam) {
  el("durum-noktasi").className = `nokta ${kaynakTamam ? "canli" : "kopuk"}`;
  el("uyari").classList.toggle("gizli", Boolean(kaynakTamam));
}

function escapeHtml(metin) {
  const kutu = document.createElement("div");
  kutu.textContent = metin;
  return kutu.innerHTML;
}

/* --- geçmiş grafiği ---------------------------------------------------- */

async function gecmisGuncelle() {
  try {
    const satirlar = await (await fetch(`/api/history?minutes=${GECMIS_DAKIKA}`)).json();
    grafigiCiz(zamanaGoreTopla(satirlar));
  } catch (hata) {
    /* bir sonraki turda tekrar denenir */
  }
}

/** Aynı zaman damgasındaki bölge sayılarını toplayarak tek seri üretir. */
function zamanaGoreTopla(satirlar) {
  const toplam = new Map();
  for (const satir of satirlar) {
    toplam.set(satir.ts, (toplam.get(satir.ts) || 0) + satir.count);
  }
  return [...toplam.entries()]
    .map(([ts, sayi]) => ({ t: new Date(ts), sayi }))
    .sort((a, b) => a.t - b.t);
}

function grafigiCiz(noktalar) {
  const canvas = el("grafik");
  const olcek = window.devicePixelRatio || 1;
  const genislik = canvas.clientWidth;
  const yukseklik = canvas.height;

  canvas.width = genislik * olcek;
  canvas.style.height = `${yukseklik}px`;
  canvas.height = yukseklik * olcek;

  const ctx = canvas.getContext("2d");
  ctx.scale(olcek, olcek);
  ctx.clearRect(0, 0, genislik, yukseklik);

  const pad = { ust: 12, sag: 12, alt: 26, sol: 36 };
  const alanG = genislik - pad.sol - pad.sag;
  const alanY = yukseklik - pad.ust - pad.alt;

  if (noktalar.length === 0) {
    ctx.fillStyle = "#8b95a5";
    ctx.font = "13px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Henüz veri yok", genislik / 2, yukseklik / 2);
    return;
  }

  const enYuksek = Math.max(1, ...noktalar.map((n) => n.sayi));
  const tavan = Math.ceil(enYuksek * 1.2);
  const ilk = noktalar[0].t.getTime();
  const son = noktalar[noktalar.length - 1].t.getTime();
  const aralik = Math.max(1, son - ilk);

  const x = (nokta) => pad.sol + ((nokta.t.getTime() - ilk) / aralik) * alanG;
  const y = (sayi) => pad.ust + alanY - (sayi / tavan) * alanY;

  // yatay ızgara ve y ekseni etiketleri
  ctx.strokeStyle = "#2a313c";
  ctx.fillStyle = "#8b95a5";
  ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.lineWidth = 1;

  const adim = Math.max(1, Math.ceil(tavan / 4));
  for (let deger = 0; deger <= tavan; deger += adim) {
    const cizgiY = Math.round(y(deger)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(pad.sol, cizgiY);
    ctx.lineTo(genislik - pad.sag, cizgiY);
    ctx.stroke();
    ctx.fillText(String(deger), pad.sol - 8, cizgiY + 4);
  }

  // dolgu
  const gradyan = ctx.createLinearGradient(0, pad.ust, 0, pad.ust + alanY);
  gradyan.addColorStop(0, "rgba(77, 163, 255, 0.35)");
  gradyan.addColorStop(1, "rgba(77, 163, 255, 0.02)");

  ctx.beginPath();
  ctx.moveTo(x(noktalar[0]), pad.ust + alanY);
  noktalar.forEach((nokta) => ctx.lineTo(x(nokta), y(nokta.sayi)));
  ctx.lineTo(x(noktalar[noktalar.length - 1]), pad.ust + alanY);
  ctx.closePath();
  ctx.fillStyle = gradyan;
  ctx.fill();

  // çizgi
  ctx.beginPath();
  noktalar.forEach((nokta, i) => {
    const px = x(nokta);
    const py = y(nokta.sayi);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.strokeStyle = "#4da3ff";
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.stroke();

  // zaman etiketleri
  ctx.fillStyle = "#8b95a5";
  ctx.textAlign = "left";
  ctx.fillText(saatBicimi.format(noktalar[0].t), pad.sol, yukseklik - 8);
  ctx.textAlign = "right";
  ctx.fillText(
    saatBicimi.format(noktalar[noktalar.length - 1].t),
    genislik - pad.sag,
    yukseklik - 8
  );
}

/* --- ısı haritası ------------------------------------------------------ */

function isiHaritasiniYenile() {
  el("isi").src = `/api/heatmap.png?t=${Date.now()}`;
}

/* --- başlangıç --------------------------------------------------------- */

function baslat() {
  const bugun = new Date().toISOString().slice(0, 10);
  el("rapor-linki").href = `/api/report?date=${bugun}&format=csv`;

  canliGuncelle();
  gecmisGuncelle();

  setInterval(canliGuncelle, CANLI_ARALIK_MS);
  setInterval(gecmisGuncelle, GECMIS_ARALIK_MS);
  setInterval(isiHaritasiniYenile, ISI_ARALIK_MS);
  window.addEventListener("resize", gecmisGuncelle);
}

document.addEventListener("DOMContentLoaded", baslat);
