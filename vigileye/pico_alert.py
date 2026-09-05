# Pico hardware is not connected yet.
# Set this to True when the Raspberry Pi Pico is actually connected.

PICO_ENABLED = False

_pico = None

PICO_PORT = "COM5"
PICO_BAUD = 115200


def get_pico():
    global _pico

    if not PICO_ENABLED:
        return None

    if _pico is None:
        try:
            import serial

            _pico = serial.Serial(
                PICO_PORT,
                PICO_BAUD,
                timeout=1
            )

            print(f"[VigilEye] Pico connected on {PICO_PORT}")

        except Exception as e:
            print(f"[VigilEye] Pico not connected: {e}")

    return _pico


def send_to_pico(state: str):
    if not PICO_ENABLED:
        return

    pico = get_pico()

    if pico is None:
        return

    try:
        pico.write((state + "\n").encode())
    except Exception as e:
        print(f"[VigilEye] Pico write failed: {e}")