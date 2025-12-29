#!/usr/bin/env python3
import json, os
import paho.mqtt.client as mqtt





def send_json_to_thingsboard(payload: dict, host: str, port: int, token: str, use_tls: bool=False):
    """
    Envía datos a ThingsBoard en dos pasos:
    1) Telemetría PLANA (claves canalX_estado) → Garantiza que el widget las reciba.
    2) Payload COMPLETO (summary, channels, bandas...) → Telemetría estructurada.
    """
    try:
        client_id = f"AudioCinemaPi-{os.uname().nodename}-{os.getpid()}"
        client = mqtt.Client(client_id=client_id, clean_session=True)
        client.username_pw_set(token)

        if use_tls:
            import ssl
            client.tls_set(cert_reqs=ssl.CERT_NONE)
            client.tls_insecure_set(True)

        client.connect(host, port, keepalive=30)
        topic = "v1/devices/me/telemetry"

        # ====================================================
        # 1️⃣ PRIMER ENVÍO → SOLO TELEMETRÍA PLANA
        # ====================================================
        plano = {k: v for k, v in payload.items() if k.startswith("canal")}

        if plano:
            client.publish(topic, json.dumps(plano), qos=1)
            print("\n🔵 DEBUG plano:", json.dumps(plano, indent=2))

            print("📡 Enviada telemetría PLANA:", plano)

        # ====================================================
        # 2️⃣ SEGUNDO ENVÍO → PAYLOAD COMPLETO
        # ====================================================

        
        client.publish(topic, json.dumps(payload), qos=1)
        print("\n🟠 DEBUG payload completo:", json.dumps(payload, indent=2))
        print("📡 Enviado payload COMPLETO")

        client.loop(timeout=2.0)
        client.disconnect()

        return True

    except Exception as e:
        print(f"Error MQTT: {e}")
        return False
