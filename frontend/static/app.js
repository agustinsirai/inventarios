let orientation = "portrait";
let enableDownload = true;

const bgVideo = document.getElementById("bgVideo");
const previewImg = document.getElementById("previewImg");
const previewEmpty = document.getElementById("previewEmpty");
const metaText = document.getElementById("metaText");

const countEl = document.getElementById("count");
const btnPortrait = document.getElementById("btnPortrait");
const btnLandscape = document.getElementById("btnLandscape");
const btnGenerate = document.getElementById("btnGenerate");

const viewer = document.getElementById("viewer");
const viewerImg = document.getElementById("viewerImg");
const viewerHud = document.getElementById("viewerHud");
const btnClose = document.getElementById("btnClose");
const btnDownload = document.getElementById("btnDownload");

function setOrientation(o){
  orientation = o;
  btnPortrait.classList.toggle("isOn", o === "portrait");
  btnLandscape.classList.toggle("isOn", o === "landscape");

  
}

btnPortrait.addEventListener("click", ()=>setOrientation("portrait"));
btnLandscape.addEventListener("click", ()=>setOrientation("landscape"));

function setBusy(isBusy){
  btnGenerate.disabled = isBusy;
  btnGenerate.textContent = isBusy ? "Generando..." : "Generar";
  previewImg.style.opacity = isBusy ? "0.4" : "1";
}

async function fetchConfig(){
  const r = await fetch("/api/config");
  const j = await r.json();
  enableDownload = !!j.enable_download;
  btnDownload.style.display = enableDownload ? "inline-flex" : "none";
}

async function generate(){
  const count = parseInt(countEl.value || "12", 10);

  setBusy(true);
  metaText.textContent = "";

  try{
    const r = await fetch("/api/generate", {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({count, orientation})
    });
    const j = await r.json();

    previewImg.src = j.preview_url + "?t=" + Date.now();
    previewImg.style.display = "block";
    previewEmpty.style.display = "none";

    // metadata mínima (sin explicación)
    metaText.textContent = j.print_id;

    // viewer wiring
    viewerImg.src = previewImg.src;
    if (enableDownload){
      btnDownload.href = j.preview_url;
      btnDownload.setAttribute("download", j.print_id + ".jpg");
    }
  }catch(e){
    metaText.textContent = "Error.";
  }finally{
    setBusy(false);
  }
}

btnGenerate.addEventListener("click", generate);

previewImg.addEventListener("click", ()=>{
  if (!previewImg.src) return;
  viewerImg.src = previewImg.src;
  viewer.classList.add("isOn");
  viewer.setAttribute("aria-hidden","false");
  viewerHud.classList.add("isHidden");
});

// tap/click en fullscreen alterna HUD
viewer.addEventListener("click", (e)=>{
  // si clickean un botón, no togglees
  if (e.target === btnClose || e.target === btnDownload) return;
  viewerHud.classList.toggle("isHidden");
});

btnClose.addEventListener("click", ()=>{
  viewer.classList.remove("isOn");
  viewer.setAttribute("aria-hidden","true");
});

// init
setOrientation("portrait");
fetchConfig();
function updateBackgroundVideo(){
  const isLandscapeScreen = window.innerWidth > window.innerHeight;

  const src = isLandscapeScreen
    ? "/static/video/01NM_1.mp4"
    : "/static/video/home-vertical.mp4";

  if (bgVideo.getAttribute("src") !== src){
    bgVideo.setAttribute("src", src);
    bgVideo.load();
    bgVideo.play().catch(()=>{});
  }
}

window.addEventListener("resize", updateBackgroundVideo);
window.addEventListener("orientationchange", updateBackgroundVideo);

updateBackgroundVideo();