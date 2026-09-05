import serial

PICO_PORT = "COM7"   # replace with YOUR actual port number
_pico = None

def get_pico():
    global _pico
    if _pico is None:
        try:
            _pico = serial.Serial(PICO_PORT, 115200, timeout=1)
        except Exception as e:
            print(f"[VigilEye] Could not open Pico serial port: {e}")
    return _pico

def send_to_pico(state: str):
    pico = get_pico()
    if pico is None:
        return
    try:
        pico.write((state + "\n").encode())
    except Exception as e:
        print(f"[VigilEye] Pico write failed: {e}")