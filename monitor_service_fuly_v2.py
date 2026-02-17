#!/usr/bin/env python3
"""
ANDROMEDA SCADA MONITOR v2.0
============================
Mejoras principales:
1. Sistema de confirmación de 60 segundos con múltiples escaneos
2. Soporte para múltiples destinatarios de correo
3. Detección mejorada de cambio de ventana/focus
4. Anti-rebote para evitar falsos positivos al cambiar entre programas
5. Estado de "cuarentena" para cambios sospechosos
"""
import cv2
import numpy as np
import time
import json
import os
import sys
import traceback
import signal
import pytz
import schedule
import csv
import threading
import logging
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
from mss import mss
import pytesseract
from flask import Flask, jsonify
from logging.handlers import RotatingFileHandler
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from telegram import Bot

CONFIG_FILE = "config.json"

with open(CONFIG_FILE) as f:
    CONFIG = json.load(f)

TURBINES_CONFIG = CONFIG["turbines"]
DETECTION_CONFIG = CONFIG["detection"]
EMAIL = CONFIG["notifications"]["email"]
TELEGRAM = CONFIG["notifications"]["telegram"]
PATHS = CONFIG["paths"]

# ========================
# 🔧 NUEVAS CONFIGURACIONES
# ========================
SCADA_TIMEOUT = 30 * 60  # 30 minutos
CONFIRMATION_TIME = 60   # Segundos para confirmar un cambio (60 segundos = múltiples escaneos)
MIN_CONFIRMATIONS = 3    # Mínimo de escaneos consistentes para confirmar cambio
STABILIZATION_TIME = 10  # Segundos de estabilización después de que SCADA vuelva

os.makedirs(os.path.dirname(PATHS["log_file"]), exist_ok=True)

logger = logging.getLogger("andromeda")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(PATHS["log_file"], maxBytes=2_000_000, backupCount=5)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Añadir también log a consola para debug
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

bot = Bot(token=TELEGRAM["token"])
app = Flask(__name__)

last_state_global = {}


class PendingChange:
    """Clase para gestionar cambios pendientes de confirmación"""
    def __init__(self, turbine, old_state, new_state):
        self.turbine = turbine
        self.old_state = old_state
        self.new_state = new_state
        self.first_seen = time.time()
        self.confirmations = 1
        self.last_confirmation = time.time()
    
    def confirm(self):
        """Añadir una confirmación del mismo estado"""
        self.confirmations += 1
        self.last_confirmation = time.time()
    
    def is_expired(self):
        """Verificar si el cambio ha expirado (estado volvió al original)"""
        return time.time() - self.last_confirmation > CONFIRMATION_TIME
    
    def is_confirmed(self):
        """Verificar si el cambio está confirmado (suficientes escaneos en 60 segundos)"""
        time_elapsed = time.time() - self.first_seen
        return (time_elapsed >= CONFIRMATION_TIME and 
                self.confirmations >= MIN_CONFIRMATIONS)
    
    def __repr__(self):
        return f"PendingChange({self.turbine}: {self.old_state}→{self.new_state}, conf={self.confirmations})"


class Monitor:
    def __init__(self):
        self.last_valid_state = None
        self.last_scada_time = None
        self.scada_was_hidden = False
        self.hidden_start_time = None
        self.running = True
        self.start_time = time.time()
        self.statistics = self.load_statistics()
        
        # ========================
        # 🔧 NUEVAS VARIABLES PARA CONFIRMACIÓN
        # ========================
        self.pending_changes = {}  # {turbine: PendingChange}
        self.stabilization_until = None  # Tiempo hasta que termine la estabilización
        self.scada_return_time = None  # Cuándo volvió SCADA
        self.consecutive_scada_detections = 0  # Contador de detecciones consecutivas de SCADA
        self.base_state_established = False  # Bandera para saber si ya se estableció el estado base
        
        logger.info("=" * 60)
        logger.info("🚀 ANDROMEDA MONITOR v2.0 - ANTI-FALSAS ALERTAS MEJORADO")
        logger.info(f"   Tiempo de confirmación: {CONFIRMATION_TIME}s")
        logger.info(f"   Mínimo de escaneos: {MIN_CONFIRMATIONS}")
        logger.info(f"   Estabilización post-SCADA: {STABILIZATION_TIME}s")
        logger.info("=" * 60)

    def signal_handler(self, sig, frame):
        logger.info("Stopping service")
        self.running = False
        self.save_statistics()
        sys.exit(0)

    def get_time(self):
        return datetime.now(pytz.timezone("Europe/Bucharest"))

    def send_email(self, subject, msg):
        """Enviar email a múltiples destinatarios (soporta lista o string único)"""
        if not EMAIL["enabled"]:
            return
        try:
            # 🔧 SOPORTE PARA MÚLTIPLES DESTINATARIOS
            recipients = EMAIL.get("recipients", [])
            if not recipients:
                # Compatibilidad con config antiguo (un solo recipient)
                recipient = EMAIL.get("recipient", "")
                if recipient:
                    recipients = [recipient]
            
            if not recipients:
                logger.warning("No hay destinatarios configurados para email")
                return
            
            m = MIMEMultipart()
            m["From"] = EMAIL["sender"]
            m["To"] = ", ".join(recipients)  # Múltiples destinatarios
            m["Subject"] = subject
            m.attach(MIMEText(msg))
            
            with smtplib.SMTP_SSL(EMAIL["smtp_server"], EMAIL["smtp_port"]) as s:
                s.login(EMAIL["sender"], EMAIL["password"])
                # Enviar a todos los destinatarios
                s.sendmail(EMAIL["sender"], recipients, m.as_string())
            
            logger.info(f"📧 Email enviado a {len(recipients)} destinatario(s): {subject}")
        except Exception as e:
            logger.error(f"Email error: {e}")

    async def send_telegram_async(self, msg):
        try:
            await bot.send_message(chat_id=TELEGRAM["chat_id"], text=msg)
        except Exception as e:
            logger.error(f"Telegram error: {e}")

    def send_telegram(self, msg):
        threading.Thread(target=lambda: asyncio.run(self.send_telegram_async(msg)), daemon=True).start()

    def detect_scada_ocr(self, img):
        """Detectar si la ventana SCADA Andromeda está visible"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config='--psm 6')
        return "andromeda" in text.lower()

    def is_in_stabilization(self):
        """Verificar si estamos en período de estabilización post-SCADA"""
        if self.stabilization_until is None:
            return False
        return time.time() < self.stabilization_until

    def detect_state(self):
        """
        🔧 ANTI-FALSAS ALERTAS v2.0:
        - Período de estabilización cuando SCADA vuelve
        - No analiza durante cambio de ventana
        - Requiere múltiples detecciones consecutivas de SCADA
        """
        now = time.time()
        
        with mss() as sct:
            img = np.array(sct.grab(sct.monitors[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        is_scada = self.detect_scada_ocr(img)

        # ========================
        # 🔧 SCADA NO VISIBLE - PAUSAR TODO
        # ========================
        if not is_scada:
            if not self.scada_was_hidden:
                self.scada_was_hidden = True
                self.hidden_start_time = now
                self.consecutive_scada_detections = 0
                self.base_state_established = False
                logger.info("🙈 SCADA OCULTO - PAUSA TOTAL (usuario cambió de ventana?)")
                logger.info("   → No se enviarán alertas hasta que SCADA sea estable")
            elif self.hidden_start_time and (now - self.hidden_start_time) > SCADA_TIMEOUT:
                logger.info("⏰ SCADA >30min oculto - ignorando")
            return None

        # ========================
        # 🔧 SCADA VISIBLE - PERO NECESITA ESTABILIZACIÓN
        # ========================
        self.last_scada_time = now
        self.consecutive_scada_detections += 1

        # Si SCADA estaba oculto y acaba de volver
        if self.scada_was_hidden:
            self.scada_return_time = now
            self.scada_was_hidden = False
            self.hidden_start_time = None
            
            # 🔧 INICIAR PERÍODO DE ESTABILIZACIÓN
            self.stabilization_until = now + STABILIZATION_TIME
            self.base_state_established = False
            
            # Limpiar cambios pendientes anteriores (ya no son válidos)
            if self.pending_changes:
                logger.info(f"🗑️ Descartando {len(self.pending_changes)} cambios pendientes antiguos")
                self.pending_changes.clear()
            
            logger.info("=" * 50)
            logger.info("👁️ SCADA HA VUELTO A SER VISIBLE")
            logger.info(f"   → Iniciando estabilización de {STABILIZATION_TIME}s")
            logger.info("   → NO se enviarán alertas durante este período")
            logger.info("=" * 50)
            return None

        # ========================
        # 🔧 PERÍODO DE ESTABILIZACIÓN ACTIVO
        # ========================
        if self.is_in_stabilization():
            remaining = self.stabilization_until - now
            logger.debug(f"⏳ Estabilización: {remaining:.1f}s restantes")
            return None

        # ========================
        # 🔧 ESTABLECER ESTADO BASE (UNA SOLA VEZ DESPUÉS DE ESTABILIZACIÓN)
        # ========================
        if not self.base_state_established:
            state = self._analyze_turbines(img)
            self.last_valid_state = state
            self.base_state_established = True
            
            logger.info("=" * 50)
            logger.info("🔄 NUEVO ESTADO BASE ESTABLECIDO (sin alertas)")
            logger.info(f"   Turbinas NO_COM: {state['nocom']}")
            logger.info(f"   Turbinas FAILURE: {state['failures']}")
            logger.info(f"   Turbinas OK: {[t for t,s in state['states'].items() if s=='OK']}")
            logger.info("   → A partir de ahora, cambios serán monitorizados")
            logger.info("   → Se requieren 60s de confirmación para alertar")
            logger.info("=" * 50)
            return state

        # ========================
        # 🔧 ANÁLISIS NORMAL (SCADA ESTABLE)
        # ========================
        return self._analyze_turbines(img)

    def _analyze_turbines(self, img):
        """Analiza colores: ROJO=FAILURE, AMARILLO=NO_COM, resto=OK"""
        failures, nocom = [], []
        states = {}

        for name, cfg in TURBINES_CONFIG.items():
            x, y, w, h = cfg["x"], cfg["y"], cfg["w"], cfg["h"]
            if y+h > img.shape[0] or x+w > img.shape[1]:
                logger.warning(f"ROI {name} fuera de pantalla")
                continue

            roi = img[y:y + h, x:x + w]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # ROJO (alarma)
            red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
            red_pct = np.sum(red_mask > 0) / (w * h) * 100

            # AMARILLO (NO_COM)
            yellow_mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
            yellow_pct = np.sum(yellow_mask > 0) / (w * h) * 100

            if red_pct > cfg["sensitivity_red"]:
                states[name] = "FAILURE"
                failures.append(name)
            elif yellow_pct > cfg["sensitivity_yellow"]:
                states[name] = "NO_COM"
                nocom.append(name)
            else:
                states[name] = "OK"

        return {
            "states": states,
            "failures": failures,
            "nocom": nocom,
            "timestamp": self.get_time().isoformat(),
        }

    def process_changes(self, new):
        """
        🔧 SISTEMA DE CONFIRMACIÓN DE CAMBIOS v2.0
        - Los cambios no se alertan inmediatamente
        - Se guardan como "pendientes" y se confirman durante 60 segundos
        - Solo si el cambio persiste durante múltiples escaneos se envía alerta
        """
        global last_state_global

        # Si no hay estado base, establecerlo sin alertar
        if not self.last_valid_state or not self.base_state_established:
            logger.info("🔄 Estableciendo estado base inicial - SIN ALERTAS")
            self.last_valid_state = new
            self.base_state_established = True
            last_state_global = new
            return

        # ========================
        # 🔧 DETECTAR CAMBIOS RESPECTO AL ÚLTIMO ESTADO VÁLIDO
        # ========================
        changes_detected = self._detect_state_changes(new)
        
        # ========================
        # 🔧 PROCESAR CAMBIOS PENDIENTES
        # ========================
        self._process_pending_changes(new, changes_detected)
        
        # ========================
        # 🔧 VERIFICAR Y ALERTAR CAMBIOS CONFIRMADOS
        # ========================
        self._alert_confirmed_changes()

        # Actualizar estadísticas y CSV
        self.update_statistics(new)
        self.export_csv(new)
        last_state_global = new

    def _detect_state_changes(self, new):
        """Detectar cambios entre el estado actual y el estado base confirmado"""
        changes = []
        
        prev_f = set(self.last_valid_state["failures"])
        prev_n = set(self.last_valid_state["nocom"])
        curr_f = set(new["failures"])
        curr_n = set(new["nocom"])

        # Nuevos FAILURE
        for t in curr_f - prev_f:
            changes.append((t, self.last_valid_state["states"].get(t, "OK"), "FAILURE"))
        
        # Nuevos NO_COM
        for t in curr_n - prev_n:
            changes.append((t, self.last_valid_state["states"].get(t, "OK"), "NO_COM"))
        
        # Recuperados de FAILURE
        for t in prev_f - curr_f:
            changes.append((t, "FAILURE", new["states"].get(t, "OK")))
        
        # Recuperados de NO_COM
        for t in prev_n - curr_n:
            changes.append((t, "NO_COM", new["states"].get(t, "OK")))
        
        return changes

    def _process_pending_changes(self, new, changes_detected):
        """Gestionar cambios pendientes de confirmación"""
        now = time.time()
        
        # Crear set de cambios actuales para búsqueda rápida
        current_changes = {(t, old, new_st) for t, old, new_st in changes_detected}
        
        # Actualizar o crear cambios pendientes
        for turbine, old_state, new_state in changes_detected:
            key = f"{turbine}:{old_state}:{new_state}"
            
            if key in self.pending_changes:
                # Ya existe - confirmar
                self.pending_changes[key].confirm()
                logger.debug(f"📊 Confirmación #{self.pending_changes[key].confirmations} para {turbine}: {old_state}→{new_state}")
            else:
                # Nuevo cambio detectado - añadir a pendientes
                self.pending_changes[key] = PendingChange(turbine, old_state, new_state)
                logger.info(f"🔍 CAMBIO DETECTADO (pendiente de confirmación): {turbine} {old_state}→{new_state}")
                logger.info(f"   → Se requieren {CONFIRMATION_TIME}s y {MIN_CONFIRMATIONS} escaneos para confirmar")
        
        # Limpiar cambios que ya no se detectan (volvieron al estado original)
        keys_to_remove = []
        for key, pending in self.pending_changes.items():
            turbine = pending.turbine
            current_state = new["states"].get(turbine, "OK")
            
            # Si el estado actual es igual al estado ORIGINAL (antes del cambio), cancelar
            if current_state == pending.old_state:
                logger.info(f"↩️ Cambio cancelado: {turbine} volvió a {pending.old_state} (no confirmado)")
                keys_to_remove.append(key)
            # Si el cambio ha expirado sin más confirmaciones
            elif pending.is_expired():
                logger.info(f"⏰ Cambio expirado: {turbine} {pending.old_state}→{pending.new_state} (sin confirmación reciente)")
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.pending_changes[key]

    def _alert_confirmed_changes(self):
        """Enviar alertas para cambios que han sido confirmados durante 60 segundos"""
        keys_to_remove = []
        
        for key, pending in self.pending_changes.items():
            if pending.is_confirmed():
                # ¡Cambio confirmado! Enviar alerta
                msg = self._format_alert_message(pending.turbine, pending.old_state, pending.new_state)
                
                logger.info("=" * 50)
                logger.info(f"✅ CAMBIO CONFIRMADO después de {CONFIRMATION_TIME}s y {pending.confirmations} escaneos")
                logger.info(f"   {pending.turbine}: {pending.old_state} → {pending.new_state}")
                logger.info("=" * 50)
                
                self.alert(msg)
                
                # Actualizar el estado válido con este cambio confirmado
                self.last_valid_state["states"][pending.turbine] = pending.new_state
                if pending.new_state == "FAILURE":
                    if pending.turbine not in self.last_valid_state["failures"]:
                        self.last_valid_state["failures"].append(pending.turbine)
                    if pending.turbine in self.last_valid_state["nocom"]:
                        self.last_valid_state["nocom"].remove(pending.turbine)
                elif pending.new_state == "NO_COM":
                    if pending.turbine not in self.last_valid_state["nocom"]:
                        self.last_valid_state["nocom"].append(pending.turbine)
                    if pending.turbine in self.last_valid_state["failures"]:
                        self.last_valid_state["failures"].remove(pending.turbine)
                else:  # OK
                    if pending.turbine in self.last_valid_state["failures"]:
                        self.last_valid_state["failures"].remove(pending.turbine)
                    if pending.turbine in self.last_valid_state["nocom"]:
                        self.last_valid_state["nocom"].remove(pending.turbine)
                
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.pending_changes[key]
        
        # Log de estado si no hay cambios confirmados
        if not keys_to_remove and not self.pending_changes:
            logger.debug(f"✅ Sistema estable: {len(self.last_valid_state['nocom'])} NO_COM, {len(self.last_valid_state['failures'])} FAILURE")

    def _format_alert_message(self, turbine, old_state, new_state):
        """Formatear mensaje de alerta según el tipo de cambio"""
        if new_state == "FAILURE":
            return f"🚨 FAILURE {turbine} (confirmado tras {CONFIRMATION_TIME}s)"
        elif new_state == "NO_COM":
            return f"📴 NO COM {turbine} (confirmado tras {CONFIRMATION_TIME}s)"
        elif old_state == "FAILURE":
            return f"✅ RECOVERED {turbine} de FAILURE (confirmado tras {CONFIRMATION_TIME}s)"
        elif old_state == "NO_COM":
            return f"📶 COM BACK {turbine} (confirmado tras {CONFIRMATION_TIME}s)"
        else:
            return f"ℹ️ {turbine}: {old_state} → {new_state}"

    def alert(self, msg):
        logger.warning(f"📢 ALERTA: {msg}")
        self.send_email(msg, msg)
        self.send_telegram(msg)

    def load_statistics(self):
        if os.path.exists(PATHS["statistics_file"]):
            with open(PATHS["statistics_file"]) as f:
                return json.load(f)
        return {t: {"ok": 0, "fail": 0, "nocom": 0} for t in TURBINES_CONFIG}

    def save_statistics(self):
        with open(PATHS["statistics_file"], "w") as f:
            json.dump(self.statistics, f, indent=2)

    def update_statistics(self, new):
        for t, st in new["states"].items():
            if t not in self.statistics:
                self.statistics[t] = {"ok": 0, "fail": 0, "nocom": 0}
            if st == "OK":
                self.statistics[t]["ok"] += 1
            elif st == "FAILURE":
                self.statistics[t]["fail"] += 1
            elif st == "NO_COM":
                self.statistics[t]["nocom"] += 1
        with open(PATHS["statistics_file"], "w") as f:
            json.dump(self.statistics, f, indent=2)

    def export_csv(self, new):
        header = ["time"] + list(TURBINES_CONFIG.keys())
        csv_file = PATHS["csv_file"]
        exists = os.path.exists(csv_file)
        with open(csv_file, "a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(header)
            row = [new["timestamp"]]
            for t in TURBINES_CONFIG:
                row.append(new["states"].get(t, ""))
            w.writerow(row)

    def daily_report(self):
        report = "📊 DAILY REPORT\n"
        for t, d in self.statistics.items():
            total = d["ok"] + d["fail"] + d["nocom"]
            if total > 0:
                av = d["ok"] / total * 100
                report += f"{t}: {av:.1f}% OK\n"
        self.alert(report)

    def watchdog(self):
        while True:
            time.sleep(60)
            if time.time() - self.start_time > 300 and not self.running:
                logger.error("Watchdog restart")
                os.execv(sys.executable, ['python'] + sys.argv)

    def run(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        schedule.every().day.at("08:00").do(self.daily_report)
        threading.Thread(target=self.watchdog, daemon=True).start()

        logger.info("🚀 Monitor anti-falsas-alertas v2.0 iniciado")
        logger.info(f"   Intervalo de escaneo: {DETECTION_CONFIG['interval']}s")

        while self.running:
            schedule.run_pending()
            state = self.detect_state()
            if state:
                self.process_changes(state)
            time.sleep(DETECTION_CONFIG["interval"])


@app.route("/api/state")
def api_state():
    return jsonify(last_state_global)


@app.route("/api/pending")
def api_pending():
    """Endpoint para ver cambios pendientes de confirmación"""
    monitor_instance = app.config.get('monitor')
    if monitor_instance:
        pending = {k: {
            'turbine': v.turbine,
            'old': v.old_state,
            'new': v.new_state,
            'confirmations': v.confirmations,
            'time_remaining': max(0, CONFIRMATION_TIME - (time.time() - v.first_seen))
        } for k, v in monitor_instance.pending_changes.items()}
        return jsonify(pending)
    return jsonify({})


@app.route("/")
def index():
    pending_html = ""
    monitor_instance = app.config.get('monitor')
    if monitor_instance and monitor_instance.pending_changes:
        pending_html = "<h2>⏳ Cambios Pendientes de Confirmación:</h2><ul>"
        for k, v in monitor_instance.pending_changes.items():
            remaining = max(0, CONFIRMATION_TIME - (time.time() - v.first_seen))
            pending_html += f"<li>{v.turbine}: {v.old_state}→{v.new_state} (confirmaciones: {v.confirmations}, restante: {remaining:.0f}s)</li>"
        pending_html += "</ul>"
    
    return f"""
    <html>
    <head>
        <title>Andromeda Monitor v2.0</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; background: #1a1a2e; color: #eee; }}
            h1 {{ color: #00d4ff; }}
            h2 {{ color: #ffd700; }}
            pre {{ background: #16213e; padding: 15px; border-radius: 8px; overflow: auto; }}
            .info {{ background: #0f3460; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>🚀 Andromeda Monitor v2.0 - Anti-Falsas Alertas</h1>
        <div class="info">
            <strong>Configuración:</strong><br>
            • Tiempo de confirmación: {CONFIRMATION_TIME}s<br>
            • Mínimo escaneos: {MIN_CONFIRMATIONS}<br>
            • Estabilización post-SCADA: {STABILIZATION_TIME}s
        </div>
        <h2>📊 Estado Actual:</h2>
        <pre>{json.dumps(last_state_global, indent=2)}</pre>
        {pending_html}
        <p><em>Auto-refresh cada 5 segundos</em></p>
    </body>
    </html>
    """


def run_web():
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    monitor = Monitor()
    app.config['monitor'] = monitor
    threading.Thread(target=run_web, daemon=True).start()
    monitor.run()
