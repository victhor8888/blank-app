#!/usr/bin/env python3
"""
ANDROMEDA SCADA MONITOR v3.0
============================
SOLUCIÓN DEFINITIVA ANTI-FALSAS ALERTAS

Mejoras v3.0:
1. Detección de SCADA por IMAGEN DE REFERENCIA (template matching) - más fiable que OCR
2. Estado "CONGELADO" cuando SCADA no está visible
3. Comparación ANTES/DESPUÉS cuando SCADA vuelve
4. Solo alerta cambios REALES confirmados durante 60 segundos
5. Soporte múltiples destinatarios email
6. Detección mejorada de colores: ROJO=alarma, AMARILLO=no_com, AZUL/GRIS/BLANCO=ok

COLORES EN SCADA:
- AMARILLO: Sin comunicación (NO_COM)
- ROJO: Alarma (FAILURE)
- AZUL/VIOLETA: OK (Running)
- GRIS/BLANCO: OK (Running)
"""
import cv2
import numpy as np
import time
import json
import os
import sys
import signal
import pytz
import schedule
import csv
import threading
import logging
import asyncio
from datetime import datetime
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
# 🔧 CONFIGURACIONES ANTI-FALSAS ALERTAS
# ========================
SCADA_TIMEOUT = 30 * 60          # 30 minutos máximo sin SCADA
CONFIRMATION_TIME = 60           # Segundos para confirmar un cambio
MIN_CONFIRMATIONS = 5            # Mínimo de escaneos consistentes (aumentado para más seguridad)
STABILIZATION_TIME = 15          # Segundos de estabilización cuando SCADA vuelve
TEMPLATE_THRESHOLD = 0.7         # Umbral para detección por imagen (0.7 = 70% similitud)
REFERENCE_IMAGE_PATH = "scada_reference.png"  # Imagen de referencia del panel lateral

os.makedirs(os.path.dirname(PATHS["log_file"]), exist_ok=True)
os.makedirs("data", exist_ok=True)

logger = logging.getLogger("andromeda")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(PATHS["log_file"], maxBytes=2_000_000, backupCount=5)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

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
        self.confirmations += 1
        self.last_confirmation = time.time()
    
    def is_expired(self):
        return time.time() - self.last_confirmation > CONFIRMATION_TIME
    
    def is_confirmed(self):
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
        # 🔧 VARIABLES PARA DETECCIÓN POR IMAGEN
        # ========================
        self.reference_template = None
        self.reference_region = None  # Región donde buscar la referencia
        self.load_reference_image()
        
        # ========================
        # 🔧 ESTADO CONGELADO (cuando SCADA no visible)
        # ========================
        self.frozen_state = None  # Estado guardado cuando SCADA desaparece
        self.pending_changes = {}
        self.stabilization_until = None
        self.base_state_established = False
        
        logger.info("=" * 70)
        logger.info("🚀 ANDROMEDA MONITOR v3.0 - DETECCIÓN POR IMAGEN")
        logger.info(f"   Tiempo de confirmación: {CONFIRMATION_TIME}s")
        logger.info(f"   Mínimo de escaneos: {MIN_CONFIRMATIONS}")
        logger.info(f"   Estabilización post-SCADA: {STABILIZATION_TIME}s")
        logger.info(f"   Umbral template matching: {TEMPLATE_THRESHOLD*100}%")
        logger.info("=" * 70)

    def load_reference_image(self):
        """Cargar imagen de referencia para detección de SCADA"""
        if os.path.exists(REFERENCE_IMAGE_PATH):
            self.reference_template = cv2.imread(REFERENCE_IMAGE_PATH)
            if self.reference_template is not None:
                logger.info(f"✅ Imagen de referencia cargada: {REFERENCE_IMAGE_PATH}")
                logger.info(f"   Tamaño: {self.reference_template.shape}")
            else:
                logger.warning(f"⚠️ No se pudo cargar: {REFERENCE_IMAGE_PATH}")
        else:
            logger.warning(f"⚠️ No existe imagen de referencia: {REFERENCE_IMAGE_PATH}")
            logger.info("   → Se usará detección combinada (OCR + panel lateral)")
            logger.info("   → Para crear referencia, ejecuta: python monitor_v3.py --create-reference")

    def create_reference_image(self):
        """Crear imagen de referencia del panel lateral de SCADA"""
        logger.info("📸 Capturando imagen de referencia...")
        logger.info("   Asegúrate de que SCADA esté visible en pantalla")
        
        time.sleep(3)  # Dar tiempo para posicionar la ventana
        
        with mss() as sct:
            img = np.array(sct.grab(sct.monitors[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Región del panel lateral izquierdo (ajustar según tu pantalla)
        # Típicamente el panel "Units" está en x=0-200, y=100-500
        panel_region = img[100:500, 0:220]
        
        cv2.imwrite(REFERENCE_IMAGE_PATH, panel_region)
        logger.info(f"✅ Imagen de referencia guardada: {REFERENCE_IMAGE_PATH}")
        
        # También guardar captura completa para referencia
        cv2.imwrite("scada_full_screenshot.png", img)
        logger.info("✅ Captura completa guardada: scada_full_screenshot.png")
        
        return panel_region

    def signal_handler(self, sig, frame):
        logger.info("Stopping service")
        self.running = False
        self.save_statistics()
        sys.exit(0)

    def get_time(self):
        return datetime.now(pytz.timezone("Europe/Bucharest"))

    def send_email(self, subject, msg):
        """Enviar email a múltiples destinatarios"""
        if not EMAIL["enabled"]:
            return
        try:
            recipients = EMAIL.get("recipients", [])
            if not recipients:
                recipient = EMAIL.get("recipient", "")
                if recipient:
                    recipients = [recipient]
            
            if not recipients:
                logger.warning("No hay destinatarios configurados para email")
                return
            
            m = MIMEMultipart()
            m["From"] = EMAIL["sender"]
            m["To"] = ", ".join(recipients)
            m["Subject"] = f"[SCADA] {subject}"
            m.attach(MIMEText(msg))
            
            with smtplib.SMTP_SSL(EMAIL["smtp_server"], EMAIL["smtp_port"]) as s:
                s.login(EMAIL["sender"], EMAIL["password"])
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

    def detect_scada_by_template(self, img):
        """
        🔧 DETECCIÓN POR IMAGEN DE REFERENCIA (Template Matching)
        Más fiable que OCR - busca el panel lateral de SCADA
        """
        if self.reference_template is None:
            return None, 0
        
        try:
            result = cv2.matchTemplate(img, self.reference_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            return max_val >= TEMPLATE_THRESHOLD, max_val
        except Exception as e:
            logger.debug(f"Template matching error: {e}")
            return None, 0

    def detect_scada_by_ocr(self, img):
        """Detección por OCR (backup)"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, config='--psm 6').lower()
            # Buscar múltiples palabras clave de SCADA
            keywords = ["andromeda", "units", "selected turbine", "average windspeed", "active production"]
            matches = sum(1 for kw in keywords if kw in text)
            return matches >= 2  # Al menos 2 coincidencias
        except Exception as e:
            logger.debug(f"OCR error: {e}")
            return False

    def detect_scada_by_panel(self, img):
        """
        🔧 DETECCIÓN POR PANEL LATERAL
        Busca el panel gris con la lista de turbinas
        """
        try:
            # El panel lateral es gris claro (RGB ~240,240,240)
            # Buscar en la región izquierda de la pantalla
            left_panel = img[100:600, 0:250]
            hsv = cv2.cvtColor(left_panel, cv2.COLOR_BGR2HSV)
            
            # Detectar gris claro (bajo saturation, alto value)
            gray_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
            gray_pct = np.sum(gray_mask > 0) / (left_panel.shape[0] * left_panel.shape[1]) * 100
            
            return gray_pct > 30  # Si más del 30% es gris claro, hay panel
        except Exception as e:
            logger.debug(f"Panel detection error: {e}")
            return False

    def is_scada_visible(self, img):
        """
        🔧 DETECCIÓN COMBINADA DE SCADA
        Usa múltiples métodos para mayor fiabilidad
        """
        methods_passed = 0
        detection_details = []
        
        # Método 1: Template matching (más fiable)
        if self.reference_template is not None:
            template_match, confidence = self.detect_scada_by_template(img)
            if template_match:
                methods_passed += 2  # Doble peso
                detection_details.append(f"Template:{confidence:.2f}")
        
        # Método 2: Detección de panel lateral
        if self.detect_scada_by_panel(img):
            methods_passed += 1
            detection_details.append("Panel:OK")
        
        # Método 3: OCR (backup)
        if self.detect_scada_by_ocr(img):
            methods_passed += 1
            detection_details.append("OCR:OK")
        
        is_visible = methods_passed >= 2
        
        if detection_details:
            logger.debug(f"Detección SCADA: {detection_details} = {is_visible}")
        
        return is_visible

    def detect_state(self):
        """
        🔧 SISTEMA ANTI-FALSAS ALERTAS v3.0
        
        Flujo:
        1. Capturar pantalla
        2. Verificar si SCADA está visible (múltiples métodos)
        3. Si NO visible: CONGELAR estado actual, no analizar
        4. Si VUELVE visible: período de estabilización, comparar con estado congelado
        5. Solo alertar cambios REALES confirmados durante 60s
        """
        now = time.time()
        
        with mss() as sct:
            img = np.array(sct.grab(sct.monitors[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        is_scada = self.is_scada_visible(img)

        # ========================
        # 🔧 SCADA NO VISIBLE - CONGELAR ESTADO
        # ========================
        if not is_scada:
            if not self.scada_was_hidden:
                # Primera detección de SCADA oculto
                self.scada_was_hidden = True
                self.hidden_start_time = now
                
                # 🔧 CONGELAR EL ESTADO ACTUAL
                if self.last_valid_state:
                    self.frozen_state = self.last_valid_state.copy()
                    logger.info("=" * 60)
                    logger.info("🧊 SCADA OCULTO - ESTADO CONGELADO")
                    logger.info(f"   Estado guardado: {len(self.frozen_state.get('nocom', []))} NO_COM, {len(self.frozen_state.get('failures', []))} FAILURE")
                    logger.info("   → NO se enviarán alertas hasta que SCADA vuelva y se estabilice")
                    logger.info("=" * 60)
                else:
                    logger.info("🙈 SCADA OCULTO - Sin estado previo para congelar")
                
                # Limpiar cambios pendientes (ya no son válidos)
                if self.pending_changes:
                    logger.info(f"🗑️ Descartando {len(self.pending_changes)} cambios pendientes")
                    self.pending_changes.clear()
                
            elif self.hidden_start_time and (now - self.hidden_start_time) > SCADA_TIMEOUT:
                logger.debug("⏰ SCADA >30min oculto")
            
            return None

        # ========================
        # 🔧 SCADA VISIBLE DE NUEVO
        # ========================
        self.last_scada_time = now

        if self.scada_was_hidden:
            # SCADA acaba de volver
            hidden_duration = now - self.hidden_start_time if self.hidden_start_time else 0
            
            logger.info("=" * 60)
            logger.info("👁️ SCADA HA VUELTO A SER VISIBLE")
            logger.info(f"   Tiempo oculto: {hidden_duration:.1f}s")
            logger.info(f"   → Iniciando estabilización de {STABILIZATION_TIME}s")
            logger.info("   → Comparando con estado congelado...")
            logger.info("=" * 60)
            
            self.scada_was_hidden = False
            self.hidden_start_time = None
            self.stabilization_until = now + STABILIZATION_TIME
            self.base_state_established = False
            
            return None

        # ========================
        # 🔧 PERÍODO DE ESTABILIZACIÓN
        # ========================
        if self.stabilization_until and now < self.stabilization_until:
            remaining = self.stabilization_until - now
            logger.debug(f"⏳ Estabilización: {remaining:.1f}s restantes")
            return None

        # ========================
        # 🔧 ESTABLECER NUEVO ESTADO BASE (comparar con congelado)
        # ========================
        if not self.base_state_established:
            current_state = self._analyze_turbines(img)
            
            logger.info("=" * 60)
            logger.info("🔄 ANALIZANDO ESTADO DESPUÉS DE ESTABILIZACIÓN")
            logger.info(f"   Estado actual: {current_state['nocom']} NO_COM, {current_state['failures']} FAILURE")
            
            if self.frozen_state:
                # Comparar con estado congelado
                frozen_nocom = set(self.frozen_state.get("nocom", []))
                frozen_fail = set(self.frozen_state.get("failures", []))
                current_nocom = set(current_state["nocom"])
                current_fail = set(current_state["failures"])
                
                # Solo hay cambio REAL si es diferente del estado congelado
                real_changes = (frozen_nocom != current_nocom) or (frozen_fail != current_fail)
                
                if real_changes:
                    logger.info("⚠️ DETECTADOS CAMBIOS REALES respecto al estado congelado:")
                    if frozen_nocom != current_nocom:
                        new_nocom = current_nocom - frozen_nocom
                        recovered_nocom = frozen_nocom - current_nocom
                        if new_nocom:
                            logger.info(f"   → Nuevos NO_COM: {new_nocom}")
                        if recovered_nocom:
                            logger.info(f"   → Recuperados de NO_COM: {recovered_nocom}")
                    if frozen_fail != current_fail:
                        new_fail = current_fail - frozen_fail
                        recovered_fail = frozen_fail - current_fail
                        if new_fail:
                            logger.info(f"   → Nuevos FAILURE: {new_fail}")
                        if recovered_fail:
                            logger.info(f"   → Recuperados de FAILURE: {recovered_fail}")
                    logger.info("   → Estos cambios requerirán confirmación de 60s")
                else:
                    logger.info("✅ Estado IDÉNTICO al congelado - NO hay cambios reales")
                    logger.info("   → No se enviarán alertas falsas")
            else:
                logger.info("   (Sin estado congelado previo)")
            
            logger.info("=" * 60)
            
            # Usar estado congelado como base si existe, sino el actual
            if self.frozen_state:
                self.last_valid_state = self.frozen_state.copy()
            else:
                self.last_valid_state = current_state
            
            self.base_state_established = True
            self.frozen_state = None  # Limpiar estado congelado
            
            return current_state

        # ========================
        # 🔧 ANÁLISIS NORMAL
        # ========================
        return self._analyze_turbines(img)

    def _analyze_turbines(self, img):
        """
        🔧 ANÁLISIS DE COLORES MEJORADO
        
        Colores en SCADA Andromeda:
        - AMARILLO: Sin comunicación (NO_COM)
        - ROJO: Alarma (FAILURE)  
        - AZUL/VIOLETA: OK (Running)
        - GRIS/BLANCO: OK (Running)
        """
        failures, nocom = [], []
        states = {}

        for name, cfg in TURBINES_CONFIG.items():
            x, y, w, h = cfg["x"], cfg["y"], cfg["w"], cfg["h"]
            if y+h > img.shape[0] or x+w > img.shape[1]:
                logger.warning(f"ROI {name} fuera de pantalla")
                continue

            roi = img[y:y + h, x:x + w]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            total_pixels = w * h

            # 🔧 ROJO (alarma) - Hue 0-10 o 170-180
            red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
            red_mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            red_pct = np.sum(red_mask > 0) / total_pixels * 100

            # 🔧 AMARILLO (NO_COM) - Hue 20-35
            yellow_mask = cv2.inRange(hsv, (20, 100, 100), (40, 255, 255))
            yellow_pct = np.sum(yellow_mask > 0) / total_pixels * 100

            # Determinar estado
            if red_pct > cfg.get("sensitivity_red", 10):
                states[name] = "FAILURE"
                failures.append(name)
            elif yellow_pct > cfg.get("sensitivity_yellow", 15):
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
        🔧 PROCESAR CAMBIOS CON CONFIRMACIÓN
        Solo alerta cambios que persisten durante 60 segundos
        """
        global last_state_global

        if not self.last_valid_state or not self.base_state_established:
            logger.info("🔄 Estableciendo estado base inicial")
            self.last_valid_state = new
            self.base_state_established = True
            last_state_global = new
            return

        # Detectar cambios
        changes_detected = self._detect_state_changes(new)
        
        # Procesar cambios pendientes
        self._process_pending_changes(new, changes_detected)
        
        # Alertar cambios confirmados
        self._alert_confirmed_changes()

        # Actualizar estadísticas
        self.update_statistics(new)
        self.export_csv(new)
        last_state_global = new

    def _detect_state_changes(self, new):
        """Detectar cambios respecto al último estado válido"""
        changes = []
        
        prev_f = set(self.last_valid_state.get("failures", []))
        prev_n = set(self.last_valid_state.get("nocom", []))
        curr_f = set(new["failures"])
        curr_n = set(new["nocom"])

        for t in curr_f - prev_f:
            changes.append((t, self.last_valid_state["states"].get(t, "OK"), "FAILURE"))
        for t in curr_n - prev_n:
            changes.append((t, self.last_valid_state["states"].get(t, "OK"), "NO_COM"))
        for t in prev_f - curr_f:
            changes.append((t, "FAILURE", new["states"].get(t, "OK")))
        for t in prev_n - curr_n:
            changes.append((t, "NO_COM", new["states"].get(t, "OK")))
        
        return changes

    def _process_pending_changes(self, new, changes_detected):
        """Gestionar cambios pendientes"""
        for turbine, old_state, new_state in changes_detected:
            key = f"{turbine}:{old_state}:{new_state}"
            
            if key in self.pending_changes:
                self.pending_changes[key].confirm()
                pc = self.pending_changes[key]
                logger.debug(f"📊 Confirmación #{pc.confirmations} para {turbine}")
            else:
                self.pending_changes[key] = PendingChange(turbine, old_state, new_state)
                logger.info(f"🔍 Cambio detectado (pendiente): {turbine} {old_state}→{new_state}")
                logger.info(f"   → Requiere {CONFIRMATION_TIME}s y {MIN_CONFIRMATIONS} escaneos")
        
        # Limpiar cambios cancelados o expirados
        keys_to_remove = []
        for key, pending in self.pending_changes.items():
            current_state = new["states"].get(pending.turbine, "OK")
            
            if current_state == pending.old_state:
                logger.info(f"↩️ Cambio cancelado: {pending.turbine} volvió a {pending.old_state}")
                keys_to_remove.append(key)
            elif pending.is_expired():
                logger.info(f"⏰ Cambio expirado: {pending.turbine}")
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.pending_changes[key]

    def _alert_confirmed_changes(self):
        """Alertar cambios confirmados (60s + múltiples escaneos)"""
        keys_to_remove = []
        
        for key, pending in self.pending_changes.items():
            if pending.is_confirmed():
                msg = self._format_alert_message(pending.turbine, pending.old_state, pending.new_state)
                
                logger.info("=" * 50)
                logger.info(f"✅ CAMBIO CONFIRMADO ({pending.confirmations} escaneos en {CONFIRMATION_TIME}s)")
                logger.info(f"   {pending.turbine}: {pending.old_state} → {pending.new_state}")
                logger.info("=" * 50)
                
                self.alert(msg)
                
                # Actualizar estado válido
                self._update_valid_state(pending)
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.pending_changes[key]

    def _update_valid_state(self, pending):
        """Actualizar estado válido después de confirmar cambio"""
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
        else:
            if pending.turbine in self.last_valid_state["failures"]:
                self.last_valid_state["failures"].remove(pending.turbine)
            if pending.turbine in self.last_valid_state["nocom"]:
                self.last_valid_state["nocom"].remove(pending.turbine)

    def _format_alert_message(self, turbine, old_state, new_state):
        """Formatear mensaje de alerta"""
        if new_state == "FAILURE":
            return f"🚨 ALARMA {turbine} (confirmado {CONFIRMATION_TIME}s)"
        elif new_state == "NO_COM":
            return f"📴 SIN COMUNICACIÓN {turbine} (confirmado {CONFIRMATION_TIME}s)"
        elif old_state == "FAILURE":
            return f"✅ RECUPERADO {turbine} de ALARMA (confirmado {CONFIRMATION_TIME}s)"
        elif old_state == "NO_COM":
            return f"📶 COMUNICACIÓN RECUPERADA {turbine} (confirmado {CONFIRMATION_TIME}s)"
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
        report = "📊 INFORME DIARIO\n"
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

        logger.info("🚀 Monitor v3.0 iniciado")
        logger.info(f"   Intervalo: {DETECTION_CONFIG['interval']}s")

        while self.running:
            schedule.run_pending()
            state = self.detect_state()
            if state:
                self.process_changes(state)
            time.sleep(DETECTION_CONFIG["interval"])


# ========================
# 🔧 WEB INTERFACE
# ========================
@app.route("/api/state")
def api_state():
    return jsonify(last_state_global)

@app.route("/api/pending")
def api_pending():
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
    monitor_instance = app.config.get('monitor')
    
    frozen_info = ""
    if monitor_instance and monitor_instance.frozen_state:
        frozen_info = f"""
        <div class="frozen">
            <h2>🧊 ESTADO CONGELADO (SCADA oculto)</h2>
            <pre>{json.dumps(monitor_instance.frozen_state, indent=2)}</pre>
        </div>
        """
    
    pending_html = ""
    if monitor_instance and monitor_instance.pending_changes:
        pending_html = "<h2>⏳ Cambios Pendientes de Confirmación:</h2><ul>"
        for k, v in monitor_instance.pending_changes.items():
            remaining = max(0, CONFIRMATION_TIME - (time.time() - v.first_seen))
            pending_html += f"<li>{v.turbine}: {v.old_state}→{v.new_state} (confirmaciones: {v.confirmations}/{MIN_CONFIRMATIONS}, restante: {remaining:.0f}s)</li>"
        pending_html += "</ul>"
    
    scada_status = "🟢 VISIBLE" if (monitor_instance and not monitor_instance.scada_was_hidden) else "🔴 OCULTO"
    
    return f"""
    <html>
    <head>
        <title>Andromeda Monitor v3.0</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; background: #1a1a2e; color: #eee; }}
            h1 {{ color: #00d4ff; }}
            h2 {{ color: #ffd700; }}
            pre {{ background: #16213e; padding: 15px; border-radius: 8px; overflow: auto; }}
            .info {{ background: #0f3460; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .frozen {{ background: #1e3a5f; border: 2px solid #00d4ff; padding: 15px; border-radius: 8px; margin: 15px 0; }}
            .status {{ font-size: 1.2em; padding: 10px; background: #0f3460; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>🚀 Andromeda Monitor v3.0 - Anti-Falsas Alertas</h1>
        <div class="status">
            <strong>Estado SCADA:</strong> {scada_status}
        </div>
        <div class="info">
            <strong>Configuración:</strong><br>
            • Tiempo de confirmación: {CONFIRMATION_TIME}s<br>
            • Mínimo escaneos: {MIN_CONFIRMATIONS}<br>
            • Estabilización post-SCADA: {STABILIZATION_TIME}s<br>
            • Detección: Template + Panel + OCR
        </div>
        {frozen_info}
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
    # Opción para crear imagen de referencia
    if len(sys.argv) > 1 and sys.argv[1] == "--create-reference":
        monitor = Monitor()
        monitor.create_reference_image()
        print("\n✅ Imagen de referencia creada. Ahora ejecuta sin argumentos.")
        sys.exit(0)
    
    monitor = Monitor()
    app.config['monitor'] = monitor
    threading.Thread(target=run_web, daemon=True).start()
    monitor.run()
