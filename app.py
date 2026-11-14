# -*- coding: utf-8 -*-
"""
CYBERVISION - Détection et prédiction de 10 cyberattaques (Streamlit)
=====================================================================
Pipeline conservé du notebook « projet_final_10_attaques » :
  fusion CSV -> doublons -> fillna(0) -> X / y (LabelEncoder) -> get_dummies
  -> split 70 / 15 / 15 (stratifié) -> StandardScaler (fit sur TRAIN)
  -> 5 modèles : XGBoost, Random Forest, SVM, Régression logistique, Régression linéaire
  -> métriques macro (Accuracy, Précision, Recall, F1) sur validation puis test.

Lancer :  streamlit run app.py
"""
import base64
import hashlib
import hmac
import io
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# --------------------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------------------
APP_NAME = "CYBERVISION"
DATA_PATH = Path(__file__).with_name("dataset.csv")

# Identifiants (stockés sous forme d'empreintes SHA-256, pas en clair)
USER_HASH = "83cbd86230ce0ba4bab818cef4aec704c0347699c1f4f911d86dfc53960011c7"
PASS_HASH = "219fd5a93130a8e9b20b1047c9c3828fa95403db890e55e08d3635a87c4e9a22"

TARGET = "attack_type"
DROP_COLS = ["attack_type", "is_attack", "sample_id", "timestamp_utc",
             "src_ip", "dst_ip", "src_port", "service"]
MODEL_ORDER = ["XGBoost", "Random Forest", "SVM", "Régression logistique", "Régression linéaire"]
SCALED_MODELS = {"SVM", "Régression logistique", "Régression linéaire"}
SVM_MAX_TRAIN = 10000          # le SVM (noyau rbf) est sous-échantillonné pour rester rapide
SOFTMAX_TEMP = 3.0             # utilisé pour convertir les scores du SVM en pseudo-probabilités

PRIMARY = "#3B5BDB"
INK = "#1B2559"
MUTED = "#6B7AA8"
GRID = "#E6ECFA"
GOOD = "#16A34A"
BAD = "#E11D48"

CLASS_COLORS = {
    "DDoS": "#4F46E5", "Phishing": "#8B5CF6", "SQL_Injection": "#2563EB",
    "Brute_Force": "#06B6D4", "XSS": "#14B8A6", "Malware": "#F59E0B",
    "Ransomware": "#EF4444", "Port_Scanning": "#EC4899",
    "Man_in_the_Middle": "#64748B", "Botnet": "#0EA5E9", "Normal": "#22C55E",
}
MONTHS_FR = ["Janv.", "Févr.", "Mars", "Avr.", "Mai", "Juin",
             "Juil.", "Août", "Sept.", "Oct.", "Nov.", "Déc."]

# Variables exposées sous forme de compteurs de vitesse : (colonne, libellé, unité)
GAUGES = [
    ("packet_rate_pps", "Taux de paquets", "pps"),
    ("flow_duration_ms", "Durée du flux", "ms"),
    ("unique_dst_ips_5m", "IP destination uniques", "sur 5 min"),
    ("unique_dst_ports_5m", "Ports destination uniques", "sur 5 min"),
    ("outbound_connections_5m", "Connexions sortantes", "sur 5 min"),
    ("http_requests_5m", "Requêtes HTTP", "sur 5 min"),
    ("http_4xx_5m", "Erreurs HTTP 4xx", "sur 5 min"),
    ("failed_logins_5m", "Tentatives échouées", "sur 5 min"),
    ("dns_queries_5m", "Requêtes DNS", "sur 5 min"),
    ("process_spawns_5m", "Processus lancés", "sur 5 min"),
    ("email_links_count", "Liens dans l'e-mail", "liens"),
    ("payload_entropy", "Entropie de la charge", "bits"),
]

PAGES = [
    ("dashboard", "Tableau de bord", "dashboard"),
    ("predict", "Prédiction / Test", "speed"),
    ("analyses", "Analyses", "bar_chart"),
    ("incidents", "Incidents", "warning"),
    ("data", "Données & Modèle", "database"),
    ("eval", "Évaluation", "fact_check"),
    ("settings", "Paramètres", "settings"),
]


def nice(c):
    """Nom d'affichage d'une classe."""
    return {"SQL_Injection": "SQL Injection", "Brute_Force": "Brute Force",
            "Man_in_the_Middle": "Man-in-the-Middle", "Port_Scanning": "Port Scanning"}.get(c, c)


def fint(n):
    return f"{int(n):,}".replace(",", "\u202f")


def fpct(x, d=2):
    return f"{x * 100:.{d}f}".replace(".", ",") + " %"


# --------------------------------------------------------------------------------------
# Icônes SVG (trait) - aucune émoticône
# --------------------------------------------------------------------------------------
_ICONS = {
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "alert": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "nodes": '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>',
    "bars": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "sliders": '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/>',
    "db": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
    "user": '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "lock": '<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
}


def icon(name, size=20, color="currentColor", sw=1.9):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" '
            f'stroke-linejoin="round">{_ICONS[name]}</svg>')


# --------------------------------------------------------------------------------------
# Style (CSS)
# --------------------------------------------------------------------------------------
CSS = """
<style>
:root{--ink:#1B2559;--muted:#6B7AA8;--primary:#3B5BDB;--line:#E3E9F7;}
html, body, .stApp{color-scheme:light;}
.stApp{background:linear-gradient(180deg,#EEF3FF 0%,#F7F9FF 38%,#FFFFFF 100%);color:var(--ink);
  font-family:"Inter","Segoe UI",system-ui,-apple-system,Roboto,Arial,sans-serif;}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{display:none !important;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.2rem;padding-bottom:2rem;max-width:1560px;}
h1,h2,h3,h4,p,span,label,div{color:inherit;}
section[data-testid="stSidebar"]{background:#FFFFFF;border-right:1px solid var(--line);}
section[data-testid="stSidebar"] .block-container{padding-top:1rem;}
[data-testid="stSidebarUserContent"]{padding-top:.4rem;}

/* cartes */
div[class*="st-key-card"]{background:#fff;border:1px solid var(--line);border-radius:16px;
  padding:16px 18px 12px 18px;box-shadow:0 4px 18px rgba(59,91,219,.06);}
.ctitle{display:flex;align-items:center;gap:8px;font-weight:700;font-size:.98rem;color:var(--ink);margin:0 0 6px 0;}
.ctitle svg{color:var(--primary);}
.csub{color:var(--muted);font-size:.78rem;margin:-2px 0 8px 0;}

/* entête */
.topbar{display:flex;align-items:center;gap:16px;padding:6px 4px 14px 4px;}
.topbar .ttl{font-size:1.55rem;font-weight:800;line-height:1.15;color:var(--ink);}
.topbar .sub{color:var(--muted);font-size:.84rem;margin-top:2px;}
.topbar .shield{width:54px;height:54px;border-radius:15px;background:linear-gradient(135deg,#3B5BDB,#6C8CFF);
  display:flex;align-items:center;justify-content:center;box-shadow:0 8px 20px rgba(59,91,219,.3);}
.chip{display:inline-flex;align-items:center;gap:8px;background:#fff;border:1px solid var(--line);
  border-radius:999px;padding:6px 14px 6px 8px;font-size:.84rem;font-weight:600;color:var(--ink);float:right;}
.chip .av{width:28px;height:28px;border-radius:50%;background:#E8EEFF;color:var(--primary);
  display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.8rem;}

/* KPI */
.kpi{display:flex;align-items:center;gap:14px;}
.kpi .ic{width:48px;height:48px;border-radius:50%;background:#E8EEFF;color:var(--primary);
  display:flex;align-items:center;justify-content:center;flex:none;}
.kpi .ic.red{background:#FFE9EE;color:#E11D48;}
.kpi .ic.green{background:#E3F8EA;color:#16A34A;}
.kpi .lb{font-size:.78rem;color:var(--muted);font-weight:600;}
.kpi .vl{font-size:1.65rem;font-weight:800;color:var(--ink);line-height:1.15;}
.kpi .sb{font-size:.72rem;color:var(--muted);}

/* métriques */
.mrow{display:flex;align-items:center;justify-content:space-between;padding:7px 4px;border-bottom:1px solid #EEF2FC;font-size:.86rem;}
.mrow:last-child{border-bottom:none;}
.mrow .dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:9px;}
.mrow b{color:var(--ink);}

/* tableau HTML */
table.cv{width:100%;border-collapse:collapse;font-size:.84rem;}
table.cv th{text-align:left;color:var(--muted);font-weight:600;font-size:.76rem;padding:8px 10px;border-bottom:1px solid var(--line);}
table.cv td{padding:9px 10px;border-bottom:1px solid #F0F3FC;color:var(--ink);}
.badge{display:inline-block;padding:3px 12px;border-radius:7px;font-size:.74rem;font-weight:700;}
.badge.atk{background:#FFE3E8;color:#E11D48;}
.badge.ok{background:#DDF7E6;color:#15803D;}

/* résultat de prédiction */
.res{border-radius:14px;padding:16px 18px;display:flex;align-items:center;gap:16px;justify-content:space-between;}
.res.atk{background:#FFF1F3;border:1px solid #FFC9D3;}
.res.ok{background:#EEFBF2;border:1px solid #BDEBCB;}
.res .k{font-size:.74rem;color:var(--muted);font-weight:600;}
.res .v{font-size:1.8rem;font-weight:800;line-height:1.1;}
.res.atk .v{color:#E11D48;} .res.ok .v{color:#15803D;}
.bar{height:7px;border-radius:6px;background:#EDEFF7;overflow:hidden;margin-top:6px;}
.bar > i{display:block;height:100%;border-radius:6px;}
.p3{display:flex;align-items:center;gap:10px;font-size:.84rem;margin:7px 0;}
.p3 .n{width:128px;font-weight:600;} .p3 .b{flex:1;} .p3 .pc{width:62px;text-align:right;color:var(--muted);font-weight:700;}
.stale{background:#FFF7E0;border:1px solid #F6DE9A;color:#8A6A00;border-radius:10px;padding:7px 12px;font-size:.8rem;margin-bottom:8px;}

/* widgets */
[data-testid="stWidgetLabel"] p{color:#3A4673 !important;font-weight:600;font-size:.82rem;}
div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="base-input"]{
  background:#fff !important;border-color:#DCE3F5 !important;border-radius:10px !important;color:var(--ink) !important;}
div[data-baseweb="select"] *, input{color:var(--ink) !important;}
.stButton button, .stFormSubmitButton button{border-radius:10px;font-weight:700;}
section[data-testid="stSidebar"] .stButton button{justify-content:flex-start;width:100%;border:none;
  border-radius:11px;padding:.55rem .9rem;font-weight:600;font-size:.92rem;}
div[class*="st-key-nav_"] button{background:transparent !important;color:#4A5680 !important;box-shadow:none !important;}
div[class*="st-key-nav_"] button:hover{background:#EEF3FF !important;color:var(--primary) !important;}
div[class*="st-key-navon_"] button{background:linear-gradient(135deg,#3B5BDB,#5F7CF0) !important;color:#fff !important;
  box-shadow:0 6px 16px rgba(59,91,219,.28) !important;}
div[class*="st-key-navon_"] button *{color:#fff !important;}
.st-key-btn_predict button, .stFormSubmitButton button{background:linear-gradient(135deg,#3B5BDB,#4F7BFF) !important;
  color:#fff !important;border:none !important;padding:.7rem 1rem;font-size:1rem;box-shadow:0 8px 20px rgba(59,91,219,.28);}
.st-key-btn_predict button *, .stFormSubmitButton button *{color:#fff !important;}
.st-key-btn_logout button{background:#F4F6FD !important;color:#4A5680 !important;border:1px solid var(--line) !important;}
.st-key-btn_more button{background:transparent !important;color:var(--primary) !important;border:none !important;font-size:.82rem;}
.brand{text-align:center;padding:4px 0 10px 0;}
.brand img{width:150px;}
.brand .nm{font-weight:900;letter-spacing:.18em;color:#06694A;font-size:.95rem;margin-top:-6px;}
.sbox{background:#F4F7FF;border:1px solid var(--line);border-radius:12px;padding:10px 12px;font-size:.8rem;color:var(--muted);}
.sbox b{color:var(--ink);}

/* login */
.login-wrap{text-align:center;padding:4px 0 6px 0;}
.login-wrap img{width:170px;}
.login-wrap h1{font-size:1.9rem;font-weight:900;letter-spacing:.16em;color:#06694A;margin:0;}
.login-wrap p{color:var(--muted);font-size:.9rem;margin:4px 0 8px 0;}
[data-testid="stMetricValue"]{color:var(--ink);}
</style>
"""

HIDE_SIDEBAR = """<style>section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"]{display:none !important;}</style>"""


# --------------------------------------------------------------------------------------
# Composant « compteurs de vitesse » (HTML / JS, aucune dépendance externe)
# --------------------------------------------------------------------------------------
SPEEDO_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:transparent;overflow:hidden}
body{font-family:"Inter","Segoe UI",system-ui,-apple-system,Roboto,Arial,sans-serif;color:#1B2559}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;padding:3px}
@media(max-width:640px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
.g{background:#fff;border:1px solid #E3E9F7;border-radius:14px;padding:9px 6px 6px;text-align:center;
   user-select:none;-webkit-user-select:none;outline:none;transition:box-shadow .15s,border-color .15s}
.g:hover{border-color:#C9D6FA}
.g:focus-visible{box-shadow:0 0 0 3px #C5D2FB;border-color:#3B5BDB}
.g.drag{border-color:#3B5BDB;box-shadow:0 6px 18px rgba(59,91,219,.18)}
.lbl{font-size:12px;font-weight:700;color:#3A4673;line-height:1.2;min-height:15px}
svg{width:100%;height:auto;display:block;touch-action:none;cursor:grab}
.g.drag svg{cursor:grabbing}
.val{font-weight:800;font-size:22px;fill:#1B2559}
.unit{font-size:10.5px;fill:#6B7AA8;font-weight:600}
.rng{font-size:9.5px;fill:#8F9BC4;font-weight:600}
</style></head>
<body><div id="root"></div>
<script>
(function(){
var NS="http://www.w3.org/2000/svg";
var CX=100, CY=92, R=70, A0=-120, A1=120;
var root=document.getElementById("root");
var nonce=null, gauges=[], values={}, els={}, commitTimer=null;

function send(type, extra){
  window.parent.postMessage(Object.assign({isStreamlitMessage:true,type:type}, extra||{}), "*");
}
function setHeight(){
  send("streamlit:setFrameHeight",{height:Math.ceil(document.body.getBoundingClientRect().height)+6});
}
function pt(r,a){var t=a*Math.PI/180;return [CX+r*Math.sin(t), CY-r*Math.cos(t)];}
function arc(a0,a1,r){
  var p0=pt(r,a0), p1=pt(r,a1), large=(a1-a0)>180?1:0;
  return "M"+p0[0].toFixed(2)+" "+p0[1].toFixed(2)+" A"+r+" "+r+" 0 "+large+" 1 "+p1[0].toFixed(2)+" "+p1[1].toFixed(2);
}
function el(tag,attrs,parent){
  var e=document.createElementNS(NS,tag);
  for(var k in attrs){e.setAttribute(k,attrs[k]);}
  if(parent){parent.appendChild(e);}
  return e;
}
function decimals(g){return g.step<1?2:0;}
function fmt(v,g){
  var d=decimals(g);
  return Number(v).toLocaleString("fr-FR",{minimumFractionDigits:d,maximumFractionDigits:d});
}
function colorFor(f){return f>0.88?"#E11D48":(f>0.66?"#F59E0B":"#3B5BDB");}

function update(g){
  var e=els[g.key], v=values[g.key];
  var f=(v-g.min)/(g.max-g.min); f=Math.max(0,Math.min(1,f));
  var ang=A0+f*(A1-A0);
  e.needle.setAttribute("transform","rotate("+ang.toFixed(2)+" "+CX+" "+CY+")");
  e.fill.setAttribute("d", f>0.002? arc(A0,ang,R) : "");
  e.fill.setAttribute("stroke",colorFor(f));
  e.val.textContent=fmt(v,g);
  e.card.setAttribute("aria-valuenow",String(v));
}
function valueFromEvent(ev,g){
  var s=els[g.key].svg, r=s.getBoundingClientRect();
  var x=(ev.clientX-r.left)*(200/r.width)-CX, y=(ev.clientY-r.top)*(158/r.height)-CY;
  var th=Math.atan2(x,-y)*180/Math.PI;
  if(th>A1){th=A1;} if(th<A0){th=A0;}
  var f=(th-A0)/(A1-A0), raw=g.min+f*(g.max-g.min);
  var v=Math.round((raw-g.min)/g.step)*g.step+g.min;
  v=Math.max(g.min,Math.min(g.max,v));
  return Number(v.toFixed(4));
}
function commit(){
  clearTimeout(commitTimer);
  send("streamlit:setComponentValue",{value:Object.assign({},values),dataType:"json"});
}
function commitSoon(){clearTimeout(commitTimer);commitTimer=setTimeout(commit,350);}

function build(){
  root.innerHTML=""; els={};
  var grid=document.createElement("div"); grid.className="grid"; root.appendChild(grid);
  gauges.forEach(function(g){
    var card=document.createElement("div"); card.className="g"; card.tabIndex=0;
    card.setAttribute("role","slider"); card.setAttribute("aria-label",g.label);
    card.setAttribute("aria-valuemin",String(g.min)); card.setAttribute("aria-valuemax",String(g.max));
    var lb=document.createElement("div"); lb.className="lbl"; lb.textContent=g.label; card.appendChild(lb);
    var svg=el("svg",{viewBox:"0 0 200 158"},null); card.appendChild(svg);
    el("path",{d:arc(A0,A1,R),fill:"none",stroke:"#E4EAFB","stroke-width":12,"stroke-linecap":"round"},svg);
    var fill=el("path",{d:"",fill:"none","stroke-width":12,"stroke-linecap":"round"},svg);
    for(var i=0;i<=20;i++){
      var a=A0+(A1-A0)*i/20, major=(i%5===0), p0=pt(R-17,a), p1=pt(R-(major?27:22),a);
      el("line",{x1:p0[0],y1:p0[1],x2:p1[0],y2:p1[1],stroke:major?"#9AA8D6":"#CBD4EE","stroke-width":major?2:1.2,"stroke-linecap":"round"},svg);
    }
    var pm=pt(R+0,A0), pM=pt(R+0,A1);
    var t0=el("text",{x:pm[0]+2,y:pm[1]+17,"text-anchor":"middle",class:"rng"},svg); t0.textContent=fmt(g.min,g);
    var t1=el("text",{x:pM[0]-2,y:pM[1]+17,"text-anchor":"middle",class:"rng"},svg); t1.textContent=fmt(g.max,g);
    var needle=el("g",{},svg);
    el("line",{x1:CX,y1:CY+8,x2:CX,y2:CY-(R-21),stroke:"#1B2559","stroke-width":3.2,"stroke-linecap":"round"},needle);
    el("circle",{cx:CX,cy:CY,r:8,fill:"#1B2559"},svg);
    el("circle",{cx:CX,cy:CY,r:3,fill:"#fff"},svg);
    var val=el("text",{x:CX,y:132,"text-anchor":"middle",class:"val"},svg);
    var un=el("text",{x:CX,y:148,"text-anchor":"middle",class:"unit"},svg); un.textContent=g.unit;
    grid.appendChild(card);
    els[g.key]={card:card,svg:svg,fill:fill,needle:needle,val:val};

    var dragging=false;
    svg.addEventListener("pointerdown",function(ev){
      dragging=true; card.classList.add("drag"); card.focus({preventScroll:true});
      try{svg.setPointerCapture(ev.pointerId);}catch(x){}
      values[g.key]=valueFromEvent(ev,g); update(g); ev.preventDefault();
    });
    svg.addEventListener("pointermove",function(ev){
      if(!dragging){return;} values[g.key]=valueFromEvent(ev,g); update(g);
    });
    function end(ev){
      if(!dragging){return;} dragging=false; card.classList.remove("drag");
      try{svg.releasePointerCapture(ev.pointerId);}catch(x){}
      commit();
    }
    svg.addEventListener("pointerup",end); svg.addEventListener("pointercancel",end);
    card.addEventListener("keydown",function(ev){
      var big=Math.max(g.step,(g.max-g.min)/40), d=0, k=ev.key;
      if(k==="ArrowRight"||k==="ArrowUp"){d=ev.shiftKey?big:g.step;}
      else if(k==="ArrowLeft"||k==="ArrowDown"){d=-(ev.shiftKey?big:g.step);}
      else if(k==="Home"){values[g.key]=g.min;}
      else if(k==="End"){values[g.key]=g.max;}
      else{return;}
      if(d!==0){values[g.key]=Number(Math.max(g.min,Math.min(g.max,values[g.key]+d)).toFixed(4));}
      update(g); commitSoon(); ev.preventDefault();
    });
    update(g);
  });
  setHeight();
}

window.addEventListener("message",function(ev){
  var d=ev.data||{};
  if(d.type!=="streamlit:render"){return;}
  var a=d.args||{};
  if(a.nonce!==nonce){
    nonce=a.nonce; gauges=a.gauges||[]; values=Object.assign({},a.values||{});
    gauges.forEach(function(g){if(values[g.key]===undefined){values[g.key]=g.min;}});
    build();
  }
});
window.addEventListener("resize",setHeight);
if(window.ResizeObserver){new ResizeObserver(setHeight).observe(document.body);}
send("streamlit:componentReady",{apiVersion:1});
})();
</script></body></html>
"""


@st.cache_resource(show_spinner=False)
def _declare_speedometer():
    folder = Path(tempfile.gettempdir()) / "cybervision_speedometer_v1"
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / "index.html"
    if not f.exists() or f.read_text(encoding="utf-8") != SPEEDO_HTML:
        f.write_text(SPEEDO_HTML, encoding="utf-8")
    return components.declare_component("cybervision_speedometer", path=str(folder))


def speedometers(specs, values, nonce, key):
    comp = _declare_speedometer()
    out = comp(gauges=specs, values=values, nonce=nonce, key=key, default=values)
    return out if isinstance(out, dict) else values


# --------------------------------------------------------------------------------------
# Données + entraînement (logique du notebook conservée)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset(raw: bytes):
    """Lit le CSV (lignes illisibles ignorées, comme dans le notebook)."""
    df = pd.read_csv(io.BytesIO(raw), on_bad_lines="skip").dropna(subset=[TARGET])
    return df, hashlib.md5(raw).hexdigest()


def softmax(z, temp=1.0):
    z = np.asarray(z, dtype=float) * temp
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def metrics_of(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Précision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def predict_with(B, name, X_df):
    """Retourne (classes prédites, matrice de probabilités ou None, scores bruts ou None)."""
    m = B["models"][name]
    Xin = B["scaler"].transform(X_df) if name in SCALED_MODELS else X_df
    n_cls = len(B["classes"])
    if name == "Régression linéaire":
        score = m.predict(Xin)
        return np.clip(np.rint(score), 0, n_cls - 1).astype(int), None, score
    if name == "SVM":
        proba = softmax(m.decision_function(Xin), SOFTMAX_TEMP)
        return m.predict(Xin), proba, None
    proba = m.predict_proba(Xin)
    return proba.argmax(axis=1), proba, None


@st.cache_resource(show_spinner=False)
def train_all(token, _df_raw):
    t_start = time.time()
    df = _df_raw
    n_raw = len(df)

    # 05 - doublons ; 06/07 - valeurs manquantes
    df = df.drop_duplicates().copy()
    n_dup = n_raw - len(df)
    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    df = df.fillna(0)

    # 08 - X et y
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(df[TARGET]), index=df.index)
    X = df.drop(columns=DROP_COLS)
    X["dst_port"] = X["dst_port"].astype(int)
    X = pd.get_dummies(X, columns=["protocol", "dst_port"], dtype=int)

    # 09 - 70 / 15 / 15 stratifié
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    # 10 - normalisation (fit sur TRAIN uniquement)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    n_cls = len(le.classes_)

    # SVM : sous-échantillon stratifié du train (vitesse)
    if len(X_train) > SVM_MAX_TRAIN:
        idx_svm, _ = train_test_split(np.arange(len(X_train)), train_size=SVM_MAX_TRAIN,
                                      random_state=42, stratify=y_train)
    else:
        idx_svm = np.arange(len(X_train))

    models, timings = {}, {}

    def fit(name, model, Xa, ya):
        t0 = time.time()
        model.fit(Xa, ya)
        models[name] = model
        timings[name] = time.time() - t0

    fit("XGBoost", XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                 eval_metric="mlogloss", random_state=42, n_jobs=2), X_train, y_train)
    fit("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2), X_train, y_train)
    fit("SVM", SVC(kernel="rbf", C=1.0), X_train_s[idx_svm], y_train.iloc[idx_svm])
    fit("Régression logistique", LogisticRegression(max_iter=1000, random_state=42), X_train_s, y_train)
    fit("Régression linéaire", LinearRegression(), X_train_s, y_train)

    B = {
        "models": models, "timings": timings, "scaler": scaler, "le": le,
        "classes": list(le.classes_), "columns": list(X.columns),
        "df": df, "n_raw": n_raw, "n_dup": n_dup, "nan_counts": nan_counts,
        "split": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "svm_train": len(idx_svm),
    }

    # 16 / 18 - métriques validation et test pour chaque modèle
    B["val"], B["test"], B["cm"], B["report"] = {}, {}, {}, {}
    B["test_pred"], B["test_conf"], B["test_index"] = {}, {}, X_test.index
    for name in MODEL_ORDER:
        p_val, _, _ = predict_with(B, name, X_val)
        p_test, proba, _ = predict_with(B, name, X_test)
        B["val"][name] = metrics_of(y_val, p_val)
        B["test"][name] = metrics_of(y_test, p_test)
        B["cm"][name] = confusion_matrix(y_test, p_test, labels=list(range(n_cls)))
        rep = classification_report(y_test, p_test, labels=list(range(n_cls)),
                                    target_names=[nice(c) for c in le.classes_],
                                    output_dict=True, zero_division=0)
        B["report"][name] = pd.DataFrame(rep).T
        B["test_pred"][name] = p_test
        B["test_conf"][name] = None if proba is None else proba.max(axis=1)
    B["y_test"] = y_test.values
    B["best"] = max(MODEL_ORDER, key=lambda n: B["val"][n]["F1-score"])

    # importances (Random Forest)
    B["importances"] = pd.Series(models["Random Forest"].feature_importances_, index=X.columns)

    # profils types par classe (médianes) pour les scénarios de test
    B["num_cols"] = [c for c in X.columns if not (c.startswith("protocol_") or c.startswith("dst_port_"))]
    B["profiles"], B["mode_proto"], B["mode_port"] = {}, {}, {}
    for cls in B["classes"]:
        sub = df[df[TARGET] == cls]
        B["profiles"][cls] = {c: float(v) for c, v in sub[B["num_cols"]].median().items()}
        B["mode_proto"][cls] = str(sub["protocol"].mode().iloc[0])
        B["mode_port"][cls] = int(sub["dst_port"].astype(int).mode().iloc[0])
    B["ports"] = sorted(int(p) for p in df["dst_port"].astype(int).unique())
    B["port_service"] = {int(p): str(s) for p, s in
                         df.assign(_p=df["dst_port"].astype(int)).groupby("_p")["service"]
                         .agg(lambda s: s.mode().iloc[0]).items()}
    B["protocols"] = sorted(df["protocol"].astype(str).unique())

    # tendance mensuelle
    ts = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    tmp = pd.DataFrame({"m": ts.dt.month, "atk": (df[TARGET] != "Normal").astype(int)}).dropna()
    B["monthly"] = tmp.groupby("m").agg(total=("atk", "size"), attacks=("atk", "sum")).reset_index()

    # spécifications des compteurs de vitesse
    B["gauge_specs"] = build_gauge_specs(B)
    B["train_seconds"] = time.time() - t_start
    return B


def _nice_ceil(x):
    if x <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(x))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if x <= m * mag:
            return float(m * mag)
    return float(10 * mag)


def _nice_step(x):
    mag = 10 ** math.floor(math.log10(max(x, 1e-9)))
    for m in (1, 2, 5, 10):
        if x <= m * mag:
            return float(m * mag)
    return float(10 * mag)


def build_gauge_specs(B):
    df = B["df"]
    specs = []
    for col, label, unit in GAUGES:
        s = pd.to_numeric(df[col], errors="coerce").fillna(0)
        top = max(float(s.quantile(0.995)), max(B["profiles"][c][col] for c in B["classes"]) * 1.05)
        vmax = _nice_ceil(top)
        is_int = bool((s % 1 == 0).all())
        step = max(1.0, _nice_step(vmax / 250)) if is_int else (0.01 if vmax <= 20 else _nice_step(vmax / 250))
        specs.append({"key": col, "label": label, "unit": unit, "min": 0.0, "max": vmax, "step": step})
    return specs


def build_row(B, numeric, protocol, port):
    row = {c: 0.0 for c in B["columns"]}
    for c, v in numeric.items():
        if c in row:
            row[c] = float(v)
    if f"protocol_{protocol}" in row:
        row[f"protocol_{protocol}"] = 1.0
    if f"dst_port_{port}" in row:
        row[f"dst_port_{port}"] = 1.0
    return pd.DataFrame([row])[B["columns"]]


def incident_frame(B, name):
    """Prédictions du jeu de test sous forme de tableau d'incidents."""
    idx = B["test_index"]
    base = B["df"].loc[idx, ["timestamp_utc", "src_ip", "dst_ip", TARGET]].copy()
    pred = np.array(B["classes"])[B["test_pred"][name]]
    conf = B["test_conf"][name]
    out = pd.DataFrame({
        "Horodatage": base["timestamp_utc"].values,
        "Source": base["src_ip"].values,
        "Destination": base["dst_ip"].values,
        "Classe réelle": base[TARGET].values,
        "Classe prédite": pred,
        "Probabilité": np.nan if conf is None else conf,
    })
    out["Résultat"] = np.where(out["Classe prédite"] == "Normal", "Normal", "Attaque")
    return out.sort_values("Horodatage", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Aides UI
# --------------------------------------------------------------------------------------
def logo_uri():
    return "data:image/png;base64," + LOGO_B64


def logo_image():
    return Image.open(io.BytesIO(base64.b64decode(LOGO_B64)))


def card(name):
    return st.container(key=f"card_{name}")


def ctitle(text, ic="activity", sub=None):
    html = f'<div class="ctitle">{icon(ic, 18)}<span>{text}</span></div>'
    if sub:
        html += f'<div class="csub">{sub}</div>'
    st.markdown(html, unsafe_allow_html=True)


def page_head(title, sub):
    st.markdown(f'<div style="margin:2px 0 14px 2px"><div style="font-size:1.35rem;font-weight:800">{title}</div>'
                f'<div style="color:{MUTED};font-size:.85rem">{sub}</div></div>', unsafe_allow_html=True)


def kpi_html(ic, label, value, sub, tone=""):
    return (f'<div class="kpi"><div class="ic {tone}">{icon(ic, 24)}</div><div>'
            f'<div class="lb">{label}</div><div class="vl">{value}</div><div class="sb">{sub}</div></div></div>')


def style_fig(fig, h=280, legend=False):
    fig.update_layout(height=h, margin=dict(l=6, r=6, t=6, b=6), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      font=dict(family="Inter, Segoe UI, sans-serif", color=INK, size=12))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def show(fig, key=None):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def table_html(headers, rows):
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="cv"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def metric_rows(m):
    cols = {"Accuracy": "#3B5BDB", "Précision": "#8B5CF6", "Recall": "#06B6D4", "F1-score": "#F59E0B"}
    return "".join(f'<div class="mrow"><span><span class="dot" style="background:{cols[k]}"></span>{k}</span>'
                   f'<b>{fpct(m[k])}</b></div>' for k in cols)


# --------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------
def page_dashboard(B, active):
    df = B["df"]
    total = len(df)
    n_atk = int((df[TARGET] != "Normal").sum())
    m = B["test"][active]

    cols = st.columns(4)
    items = [
        ("shield", "Trafic analysé", fint(total), f"{fint(B['n_dup'])} doublon(s) supprimé(s)", ""),
        ("alert", "Attaques identifiées", fint(n_atk), "10 types d'attaque", "red"),
        ("target", "Taux d'attaque", fpct(n_atk / total), "Part des flux malveillants", ""),
        ("nodes", "Précision du modèle", fpct(m["Accuracy"]), f"Accuracy (test) - {active}", "green"),
    ]
    for col, it in zip(cols, items):
        with col:
            with card(f"kpi_{it[0]}"):
                st.markdown(kpi_html(*it), unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        with card("trend"):
            ctitle("Tendances du trafic réseau", "activity", "Flux par mois (jeu de données)")
            mo = B["monthly"]
            xl = [MONTHS_FR[int(i) - 1] for i in mo["m"]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xl, y=mo["total"], name="Trafic total", mode="lines+markers",
                                     line=dict(color=PRIMARY, width=2.5), fill="tozeroy",
                                     fillcolor="rgba(59,91,219,.10)"))
            fig.add_trace(go.Scatter(x=xl, y=mo["attacks"], name="Trafic d'attaque", mode="lines+markers",
                                     line=dict(color="#8B5CF6", width=2.5)))
            style_fig(fig, 270, legend=True)
            fig.update_layout(legend=dict(orientation="h", y=1.12, x=0))
            show(fig, "dash_trend")
    with c2:
        with card("perf"):
            ctitle("Performances du modèle", "cpu", f"{active} - jeu de test")
            a, b = st.columns([1, 1])
            with a:
                fig = go.Figure(go.Pie(values=[m["Accuracy"], 1 - m["Accuracy"]], hole=0.76, sort=False,
                                       marker=dict(colors=[PRIMARY, "#E6ECFA"]), textinfo="none",
                                       hoverinfo="skip", direction="clockwise"))
                fig.add_annotation(text=f"<b>{fpct(m['Accuracy'], 1)}</b><br><span style='font-size:11px;color:{MUTED}'>Accuracy</span>",
                                   showarrow=False, font=dict(size=18, color=INK))
                style_fig(fig, 215)
                show(fig, "dash_perf")
            with b:
                st.markdown(metric_rows(m), unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    atk_counts = df.loc[df[TARGET] != "Normal", TARGET].value_counts()
    with c3:
        with card("freq"):
            ctitle("Fréquence des attaques par classe", "bars")
            s = atk_counts.sort_values(ascending=True)
            fig = go.Figure(go.Bar(x=s.values, y=[nice(c) for c in s.index], orientation="h",
                                   marker=dict(color=[CLASS_COLORS[c] for c in s.index]),
                                   text=[fint(v) for v in s.values], textposition="outside", cliponaxis=False))
            style_fig(fig, 330)
            fig.update_xaxes(visible=False)
            fig.update_layout(margin=dict(l=6, r=50, t=6, b=6))
            show(fig, "dash_freq")
    with c4:
        with card("split"):
            ctitle("Répartition des attaques (10 classes)", "nodes")
            fig = go.Figure(go.Pie(labels=[nice(c) for c in atk_counts.index], values=atk_counts.values,
                                   hole=0.62, sort=False, marker=dict(colors=[CLASS_COLORS[c] for c in atk_counts.index]),
                                   textinfo="percent", textfont=dict(size=11, color="#fff")))
            fig.add_annotation(text=f"<b>{fint(n_atk)}</b><br><span style='font-size:11px;color:{MUTED}'>attaques</span>",
                               showarrow=False, font=dict(size=18, color=INK))
            style_fig(fig, 330, legend=True)
            fig.update_layout(legend=dict(orientation="v", x=1.0, y=0.5))
            show(fig, "dash_split")

    with card("inc"):
        ctitle("Incidents récents (prédictions)", "alert", f"Derniers flux classés comme attaque par {active} sur le jeu de test")
        inc = incident_frame(B, active)
        inc = inc[inc["Résultat"] == "Attaque"].head(6)
        rows = []
        for i, r in enumerate(inc.itertuples(index=False), 1):
            prob = "-" if pd.isna(r[5]) else fpct(r[5])
            rows.append([i, str(r[0]).replace("T", " "), r[1], r[2], nice(r[4]), prob, '<span class="badge atk">Attaque</span>'])
        st.markdown(table_html(["#", "Horodatage", "Adresse source", "Adresse destination", "Classe prédite",
                                "Probabilité", "Résultat"], rows), unsafe_allow_html=True)
        st.button("Voir tout", key="btn_more", on_click=lambda: st.session_state.update(page="incidents"))


SCEN_FMT = lambda c: "Trafic normal (référence)" if c == "Normal" else nice(c)


def init_predict_state(B):
    ss = st.session_state
    if "gauges" not in ss:
        ss.nonce = 0
        ss.scenario = "Normal"
        ss.last_result = None
        ss.test_log = []
        apply_scenario(B)


def apply_scenario(B):
    ss = st.session_state
    cls = ss.get("scenario", "Normal")
    prof = B["profiles"][cls]
    ss.gauges = {s["key"]: float(min(prof[s["key"]], s["max"])) for s in B["gauge_specs"]}
    ss.base_profile = dict(prof)
    ss.protocol = B["mode_proto"][cls]
    ss.dst_port = B["mode_port"][cls]
    ss.nonce = ss.get("nonce", 0) + 1
    ss.last_result = None


def result_html(res):
    atk = res["is_attack"]
    tone = "atk" if atk else "ok"
    color = BAD if atk else GOOD
    word = "Attaque" if atk else "Trafic normal"
    if res["conf"] is not None:
        right = (f'<div class="k">Confiance</div><div class="v" style="font-size:1.7rem">{fpct(res["conf"])}</div>'
                 f'<div class="bar"><i style="width:{res["conf"] * 100:.1f}%;background:{color}"></i></div>')
    else:
        right = (f'<div class="k">Score brut</div><div class="v" style="font-size:1.7rem">{res["score"]:.2f}</div>'
                 f'<div class="k" style="margin-top:4px">arrondi en classe</div>')
    html = (f'<div class="res {tone}"><div style="display:flex;align-items:center;gap:14px">'
            f'{icon("shield" if atk else "lock", 44, color, 1.7)}<div><div class="k">Résultat de la prédiction</div>'
            f'<div class="v">{word}</div></div></div><div style="min-width:140px">{right}</div></div>')
    cls = res["cls"]
    html += (f'<div style="margin:14px 2px 4px 2px;display:flex;align-items:center;gap:10px">'
             f'<span class="k" style="color:{MUTED};font-weight:600;font-size:.8rem">Classe prédite</span>'
             f'<span class="badge" style="background:{CLASS_COLORS[cls]}1F;color:{CLASS_COLORS[cls]};font-size:.85rem">{nice(cls)}</span></div>')
    if res["top3"]:
        html += f'<div class="k" style="color:{MUTED};font-weight:600;font-size:.8rem;margin:10px 2px 2px">Probabilités (Top 3)</div>'
        for name, p in res["top3"]:
            html += (f'<div class="p3"><span class="n">{nice(name)}</span><span class="b"><div class="bar" style="margin:0">'
                     f'<i style="width:{max(p * 100, 0.8):.1f}%;background:{CLASS_COLORS[name]}"></i></div></span>'
                     f'<span class="pc">{fpct(p)}</span></div>')
    else:
        html += f'<div class="csub" style="margin-top:8px">La régression linéaire ne fournit pas de probabilités.</div>'
    return html


def page_predict(B, active):
    init_predict_state(B)
    ss = st.session_state
    left, right = st.columns([2.05, 1])

    with left:
        with card("params"):
            ctitle("Paramètres du trafic", "sliders",
                   "Faites glisser l'aiguille de chaque compteur (ou utilisez les flèches du clavier).")
            a, b, c = st.columns(3)
            a.selectbox("Profil de trafic", ["Normal"] + [x for x in B["classes"] if x != "Normal"],
                        key="scenario", format_func=SCEN_FMT, on_change=apply_scenario, args=(B,))
            b.selectbox("Protocole", B["protocols"], key="protocol")
            c.selectbox("Port de destination", B["ports"], key="dst_port",
                        format_func=lambda p: f"{p}  ({B['port_service'].get(p, '')})")
            vals = speedometers(B["gauge_specs"], ss.gauges, ss.nonce, key=f"speedo_{ss.nonce}")
            ss.gauges = vals

    numeric = dict(ss.base_profile)
    numeric.update({k: float(v) for k, v in vals.items()})
    sig = hashlib.md5(json.dumps([active, ss.protocol, ss.dst_port, ss.scenario, sorted(numeric.items())],
                                 default=str).encode()).hexdigest()

    with right:
        with card("test"):
            ctitle("Partie de test", "cpu", f"Modèle actif : <b>{active}</b> (modifiable dans le menu latéral)")
            log_it = st.toggle("Journaliser le test", value=True, key="log_toggle")
            if st.button("Prédire", key="btn_predict", width="stretch"):
                row = build_row(B, numeric, ss.protocol, ss.dst_port)
                idx, proba, score = predict_with(B, active, row)
                cls = B["classes"][int(idx[0])]
                top3, conf = [], None
                if proba is not None:
                    order = np.argsort(proba[0])[::-1][:3]
                    top3 = [(B["classes"][i], float(proba[0][i])) for i in order]
                    conf = float(proba[0].max())
                ss.last_result = {"cls": cls, "is_attack": cls != "Normal", "conf": conf, "top3": top3,
                                  "score": None if score is None else float(score[0]), "sig": sig, "model": active}
                if log_it:
                    ss.test_log.insert(0, {
                        "Heure": time.strftime("%Y-%m-%d %H:%M:%S"), "Modèle": active, "Profil": SCEN_FMT(ss.scenario),
                        "Protocole": ss.protocol, "Port": ss.dst_port, "Classe prédite": nice(cls),
                        "Résultat": "Attaque" if cls != "Normal" else "Normal",
                        "Confiance": None if conf is None else round(conf * 100, 2)})
            res = ss.get("last_result")
            if res:
                if res["sig"] != sig:
                    st.markdown('<div class="stale">Les paramètres ont changé depuis cette prédiction. Relancez « Prédire ».</div>',
                                unsafe_allow_html=True)
                st.markdown(result_html(res), unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="csub" style="margin-top:10px">Réglez les compteurs puis cliquez sur « Prédire ».</div>',
                            unsafe_allow_html=True)


def page_analyses(B, active):
    page_head("Analyses", "Exploration du jeu de données et des variables les plus discriminantes")
    df = B["df"]
    t1, t2, t3, t4 = st.tabs(["Répartition", "Variables importantes", "Profils par attaque", "Distributions"])

    with t1:
        c1, c2 = st.columns(2)
        with c1:
            with card("an_classes"):
                ctitle("Nombre de flux par classe", "bars", "Déséquilibre des classes (Normal inclus)")
                s = df[TARGET].value_counts().sort_values()
                fig = go.Figure(go.Bar(x=s.values, y=[nice(c) for c in s.index], orientation="h",
                                       marker=dict(color=[CLASS_COLORS[c] for c in s.index]),
                                       text=[fint(v) for v in s.values], textposition="outside", cliponaxis=False))
                style_fig(fig, 380)
                fig.update_xaxes(visible=False)
                fig.update_layout(margin=dict(l=6, r=60, t=6, b=6))
                show(fig, "an1")
        with c2:
            with card("an_proto"):
                ctitle("Protocole par classe", "nodes", "Part de TCP / UDP dans chaque classe (%)")
                ct = pd.crosstab(df[TARGET], df["protocol"], normalize="index") * 100
                fig = go.Figure()
                for i, p in enumerate(ct.columns):
                    fig.add_trace(go.Bar(name=p, x=[nice(c) for c in ct.index], y=ct[p],
                                         marker_color=[PRIMARY, "#06B6D4", "#F59E0B"][i % 3]))
                style_fig(fig, 380, legend=True)
                fig.update_layout(barmode="stack", legend=dict(orientation="h", y=1.1, x=0))
                fig.update_xaxes(tickangle=-35)
                show(fig, "an2")

    with t2:
        with card("an_imp"):
            ctitle("Importance des variables (Random Forest)", "cpu", "15 variables les plus influentes")
            imp = B["importances"].sort_values(ascending=False).head(15).sort_values()
            fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker=dict(color=PRIMARY)))
            style_fig(fig, 460)
            fig.update_layout(margin=dict(l=6, r=10, t=6, b=6))
            show(fig, "an3")

    with t3:
        with card("an_heat"):
            ctitle("Profil moyen standardisé par classe", "activity",
                   "Écart à la moyenne globale, en écarts-types (rouge : valeur élevée, bleu : valeur basse)")
            feats = [c for c in B["importances"].sort_values(ascending=False).index if c in B["num_cols"]][:14]
            num = df[feats].apply(pd.to_numeric, errors="coerce")
            z = (num.groupby(df[TARGET]).mean() - num.mean()) / num.std()
            fig = go.Figure(go.Heatmap(z=z.values, x=feats, y=[nice(c) for c in z.index], zmid=0,
                                       colorscale="RdBu_r", zmin=-3, zmax=3, colorbar=dict(thickness=10)))
            style_fig(fig, 430)
            fig.update_xaxes(tickangle=-40)
            show(fig, "an4")

    with t4:
        with card("an_box"):
            ctitle("Distribution d'une variable par classe", "bars")
            feat = st.selectbox("Variable", B["num_cols"], index=B["num_cols"].index("packet_rate_pps")
                                if "packet_rate_pps" in B["num_cols"] else 0)
            samp = df.sample(min(len(df), 25000), random_state=1)
            fig = go.Figure()
            for cls in B["classes"]:
                s = pd.to_numeric(samp.loc[samp[TARGET] == cls, feat], errors="coerce")
                fig.add_trace(go.Box(y=s, name=nice(cls), marker_color=CLASS_COLORS[cls], boxpoints=False))
            style_fig(fig, 400)
            fig.update_xaxes(tickangle=-35)
            show(fig, "an5")


def page_incidents(B, active):
    page_head("Incidents", f"Flux du jeu de test classés par {active}")
    inc = incident_frame(B, active)
    with card("inc_filters"):
        ctitle("Filtres", "sliders")
        a, b = st.columns([2, 1])
        classes = a.multiselect("Classes prédites", [c for c in B["classes"]], default=[c for c in B["classes"] if c != "Normal"],
                                format_func=nice)
        pmin = b.select_slider("Probabilité minimale", options=[0, 50, 70, 80, 90, 95, 99], value=0,
                               format_func=lambda v: f"{v} %", disabled=B["test_conf"][active] is None)
    sub = inc[inc["Classe prédite"].isin(classes)]
    if B["test_conf"][active] is not None:
        sub = sub[sub["Probabilité"] >= pmin / 100]
    with card("inc_table"):
        ctitle(f"{fint(len(sub))} incident(s)", "alert")
        view = sub.copy()
        view["Classe réelle"] = view["Classe réelle"].map(nice)
        view["Classe prédite"] = view["Classe prédite"].map(nice)
        view["Probabilité"] = view["Probabilité"] * 100
        st.dataframe(view.head(500), width="stretch", hide_index=True, height=420,
                     column_config={"Probabilité": st.column_config.ProgressColumn("Probabilité", min_value=0, max_value=100, format="%.2f %%")})
        st.caption("Affichage limité aux 500 flux les plus récents.")
    with card("inc_log"):
        ctitle("Journal des tests de scénarios", "sliders", "Prédictions lancées depuis la page « Prédiction / Test »")
        log = st.session_state.get("test_log", [])
        if log:
            st.dataframe(pd.DataFrame(log), width="stretch", hide_index=True)
            st.button("Vider le journal", key="clear_log", on_click=lambda: st.session_state.update(test_log=[]))
        else:
            st.markdown(f'<div class="csub">Aucun test enregistré pour le moment.</div>', unsafe_allow_html=True)


def page_data(B, active):
    page_head("Données & Modèle", "Chargement du CSV, prétraitement et entraînement")
    df = B["df"]
    cols = st.columns(4)
    items = [("db", "Lignes lues", fint(B["n_raw"]), "dataset.csv", ""),
             ("shield", "Lignes conservées", fint(len(df)), f"{fint(B['n_dup'])} doublon(s)", ""),
             ("bars", "Colonnes", str(df.shape[1]), f"{len(B['columns'])} variables après encodage", ""),
             ("alert", "Valeurs manquantes", fint(int(B["nan_counts"].sum())), "remplacées par 0", "red")]
    for col, it in zip(cols, items):
        with col:
            with card(f"d_kpi_{it[0]}"):
                st.markdown(kpi_html(*it), unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        with card("d_steps"):
            ctitle("Chaîne de prétraitement", "sliders")
            steps = [
                "Lecture du CSV (lignes illisibles ignorées)",
                "Suppression des doublons (avant le découpage)",
                "Valeurs manquantes remplacées par 0 (fillna)",
                "Cible : attack_type, encodée avec LabelEncoder (11 classes)",
                "Colonnes exclues : sample_id, timestamp_utc, src_ip, dst_ip, src_port, service, is_attack",
                "get_dummies sur protocol et dst_port",
                f"Découpage stratifié 70 / 15 / 15 : train {fint(B['split']['train'])}, "
                f"validation {fint(B['split']['val'])}, test {fint(B['split']['test'])}",
                "StandardScaler ajusté sur le train uniquement",
            ]
            st.markdown("".join(f'<div class="mrow"><span>{i}. {s}</span></div>' for i, s in enumerate(steps, 1)),
                        unsafe_allow_html=True)
    with c2:
        with card("d_models"):
            ctitle("Modèles entraînés", "cpu", f"Entraînement total : {B['train_seconds']:.0f} s (mis en cache)")
            desc = {"XGBoost": "100 arbres, profondeur 4, taux 0,1",
                    "Random Forest": "100 arbres",
                    "SVM": f"noyau rbf, C=1 ({fint(B['svm_train'])} lignes d'entraînement)",
                    "Régression logistique": "max_iter=1000",
                    "Régression linéaire": "baseline pédagogique (sortie arrondie)"}
            rows = [[n + (' <span class="badge ok">meilleur F1</span>' if n == B["best"] else ""),
                     desc[n], f"{B['timings'][n]:.1f} s"] for n in MODEL_ORDER]
            st.markdown(table_html(["Modèle", "Configuration", "Temps"], rows), unsafe_allow_html=True)
    with card("d_head"):
        ctitle("Aperçu des données", "db")
        st.dataframe(df.head(10), width="stretch", hide_index=True)
        if len(B["nan_counts"]):
            st.caption("Valeurs manquantes par colonne : " + ", ".join(f"{k} ({v})" for k, v in B["nan_counts"].items()))


def page_eval(B, active):
    page_head("Évaluation", f"Modèle actif : {active} - moyennes macro sur 11 classes")
    mt, mv = B["test"][active], B["val"][active]
    cols = st.columns(4)
    for col, (k, ic) in zip(cols, [("Accuracy", "target"), ("Précision", "shield"), ("Recall", "alert"), ("F1-score", "activity")]):
        with col:
            with card(f"e_kpi_{k}"):
                st.markdown(kpi_html(ic, k, fpct(mt[k]), f"Validation : {fpct(mv[k])}", "green" if k == "F1-score" else ""),
                            unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        with card("e_cmp"):
            ctitle("Comparaison des 5 modèles", "bars", "Validation - triés par F1-score")
            order = sorted(MODEL_ORDER, key=lambda n: B["val"][n]["F1-score"], reverse=True)
            fig = go.Figure()
            for k, col in zip(["Accuracy", "Précision", "Recall", "F1-score"], ["#3B5BDB", "#8B5CF6", "#06B6D4", "#F59E0B"]):
                fig.add_trace(go.Bar(name=k, x=order, y=[B["val"][n][k] for n in order], marker_color=col))
            style_fig(fig, 330, legend=True)
            fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.14, x=0))
            fig.update_yaxes(range=[0, 1.02])
            show(fig, "ev1")
    with c2:
        with card("e_tab"):
            ctitle("Tableau des métriques", "db", "Validation | Test")
            rows = []
            for n in order:
                v, t = B["val"][n], B["test"][n]
                rows.append([f"<b>{n}</b>" if n == active else n] +
                            [f"{v[k] * 100:.2f}% | {t[k] * 100:.2f}%".replace(".", ",") for k in ["Accuracy", "Précision", "Recall", "F1-score"]])
            st.markdown(table_html(["Modèle", "Accuracy", "Précision", "Recall", "F1-score"], rows), unsafe_allow_html=True)
            if "Régression linéaire" in B["val"]:
                st.markdown('<div class="csub" style="margin-top:10px">La régression linéaire est une baseline pédagogique : '
                            'les 11 classes n\'ont pas d\'ordre naturel, ses scores sont donc nettement plus faibles.</div>',
                            unsafe_allow_html=True)

    with card("e_cm"):
        ctitle("Matrice de confusion (jeu de test)", "target", "Lignes : classe réelle - colonnes : classe prédite (% par ligne)")
        cm = B["cm"][active]
        pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
        labs = [nice(c) for c in B["classes"]]
        fig = go.Figure(go.Heatmap(z=pct, x=labs, y=labs, colorscale="Blues", zmin=0, zmax=100,
                                   text=cm, texttemplate="%{text}", hovertemplate="Réel %{y}<br>Prédit %{x}<br>%{z:.1f} %<extra></extra>",
                                   colorbar=dict(thickness=10)))
        style_fig(fig, 520)
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(tickangle=-40)
        show(fig, "ev2")
    with card("e_rep"):
        ctitle("Rapport par classe (test)", "db")
        rep = B["report"][active].copy()
        rep = rep.drop(index=["accuracy"], errors="ignore")
        rep["support"] = rep["support"].astype(int)
        rep = rep.rename(columns={"precision": "Précision", "recall": "Recall", "f1-score": "F1-score", "support": "Effectif"})
        st.dataframe(rep.style.format({"Précision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}"}),
                     width="stretch")


def page_settings(B, active):
    page_head("Paramètres", "Source des données, entraînement et compte")
    with card("s_data"):
        ctitle("Jeu de données", "db", "Par défaut : dataset.csv à côté de app.py. Vous pouvez charger un autre CSV de même structure.")
        up = st.file_uploader("Remplacer le dataset (CSV)", type=["csv"])
        if up is not None and st.button("Utiliser ce fichier", key="btn_use_file"):
            st.session_state.upload_bytes = up.getvalue()
            st.session_state.pop("gauges", None)
            st.rerun()
        if st.session_state.get("upload_bytes"):
            st.info("Un fichier chargé manuellement est utilisé.")
            if st.button("Revenir au dataset.csv par défaut", key="btn_reset_file"):
                st.session_state.pop("upload_bytes", None)
                st.session_state.pop("gauges", None)
                st.rerun()
    with card("s_train"):
        ctitle("Entraînement", "cpu")
        st.markdown(f'<div class="csub">Les modèles sont entraînés une seule fois puis conservés en mémoire. '
                    f'Le SVM utilise un sous-échantillon de {fint(B["svm_train"])} lignes, et sa « confiance » est un score de décision '
                    f'normalisé (softmax), pas une probabilité calibrée.</div>', unsafe_allow_html=True)
        if st.button("Réentraîner tous les modèles", key="btn_retrain"):
            train_all.clear()
            st.session_state.pop("gauges", None)
            st.rerun()
    with card("s_acc"):
        ctitle("Compte", "user")
        st.markdown(f'<div class="csub">Connecté en tant que <b>{st.session_state.get("user", "")}</b>.</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Connexion / navigation
# --------------------------------------------------------------------------------------
def login_page():
    st.markdown(HIDE_SIDEBAR, unsafe_allow_html=True)
    ss = st.session_state
    _, mid, _ = st.columns([1, 1.15, 1])
    with mid:
        st.markdown("<div style='height:4vh'></div>", unsafe_allow_html=True)
        with card("login"):
            st.markdown(f'<div class="login-wrap"><img src="{logo_uri()}"><h1>{APP_NAME}</h1>'
                        f'<p>Détection et prédiction des cyberattaques</p></div>', unsafe_allow_html=True)
            locked = time.time() < ss.get("lock_until", 0)
            with st.form("login_form", clear_on_submit=False):
                u = st.text_input("Identifiant", placeholder="Nom d'utilisateur")
                p = st.text_input("Mot de passe", type="password", placeholder="Mot de passe")
                ok = st.form_submit_button("Se connecter", width="stretch", disabled=locked)
            if locked:
                st.error(f"Trop de tentatives. Réessayez dans {int(ss.lock_until - time.time()) + 1} s.")
            elif ok:
                good_u = hmac.compare_digest(hashlib.sha256(u.strip().encode()).hexdigest(), USER_HASH)
                good_p = hmac.compare_digest(hashlib.sha256(p.encode()).hexdigest(), PASS_HASH)
                if good_u and good_p:
                    ss.auth, ss.user, ss.fails = True, u.strip(), 0
                    ss.page = "dashboard"
                    st.rerun()
                else:
                    ss.fails = ss.get("fails", 0) + 1
                    if ss.fails >= 5:
                        ss.lock_until, ss.fails = time.time() + 30, 0
                    st.error("Identifiant ou mot de passe incorrect.")
            st.markdown(f'<div class="csub" style="text-align:center;margin-top:8px">{APP_NAME} ANALYTICS</div>',
                        unsafe_allow_html=True)


def logout():
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def sidebar(B):
    ss = st.session_state
    with st.sidebar:
        st.markdown(f'<div class="brand"><img src="{logo_uri()}"></div>', unsafe_allow_html=True)
        for pid, label, ic in PAGES:
            active_page = ss.page == pid
            st.button(label, key=f"{'navon' if active_page else 'nav'}_{pid}", icon=f":material/{ic}:",
                      width="stretch", on_click=lambda p=pid: ss.update(page=p))
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        if "active_model" not in ss or ss.active_model not in MODEL_ORDER:
            ss.active_model = B["best"]
        st.selectbox("Modèle actif", MODEL_ORDER, key="active_model")
        st.markdown(f'<div class="sbox">Meilleur F1 (validation) : <b>{B["best"]}</b></div>', unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.button("Se déconnecter", key="btn_logout", icon=":material/logout:", width="stretch", on_click=logout)


def topbar():
    user = st.session_state.get("user", "")
    st.markdown(
        f'<div class="topbar" style="justify-content:space-between"><div style="display:flex;align-items:center;gap:16px">'
        f'<div class="shield">{icon("shield", 30, "#fff", 1.8)}</div><div><div class="ttl">Prédiction des attaques classiques en cyber</div>'
        f'<div class="sub">Analyse du trafic &nbsp;•&nbsp; Prédiction des attaques &nbsp;•&nbsp; Évaluation du modèle &nbsp;•&nbsp; Test de scénarios</div></div></div>'
        f'<div class="chip" style="float:none"><span class="av">{user[:1].upper()}</span>{user}</div></div>',
        unsafe_allow_html=True)


def get_source():
    up = st.session_state.get("upload_bytes")
    if up:
        return up
    if DATA_PATH.exists():
        return DATA_PATH.read_bytes()
    return None


PAGE_FUNCS = {"dashboard": page_dashboard, "predict": page_predict, "analyses": page_analyses,
              "incidents": page_incidents, "data": page_data, "eval": page_eval, "settings": page_settings}


def main():
    st.set_page_config(page_title=f"{APP_NAME} - Prédiction des cyberattaques", page_icon=logo_image(),
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    ss = st.session_state
    ss.setdefault("auth", False)
    ss.setdefault("page", "dashboard")

    if not ss.auth:
        login_page()
        return

    raw = get_source()
    if raw is None:
        st.error("Fichier dataset.csv introuvable à côté de app.py. Chargez un CSV ci-dessous.")
        up = st.file_uploader("Dataset (CSV)", type=["csv"])
        if up is not None:
            ss.upload_bytes = up.getvalue()
            st.rerun()
        return

    with st.spinner("Chargement du CSV et entraînement des 5 modèles (première ouverture uniquement)..."):
        df_raw, token = load_dataset(raw)
        required = set(DROP_COLS) | {"protocol", "dst_port"} | {g[0] for g in GAUGES}
        missing = sorted(required - set(df_raw.columns))
        if missing:
            ss.pop("upload_bytes", None)
            st.error("Colonnes manquantes dans le CSV : " + ", ".join(missing))
            return
        B = train_all(token, df_raw)

    sidebar(B)
    topbar()
    PAGE_FUNCS[ss.page](B, ss.active_model)


# --------------------------------------------------------------------------------------
# Logo CYBERVISION ANALYTICS (PNG encodé en base64)
# --------------------------------------------------------------------------------------
LOGO_B64 = "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"

main()
