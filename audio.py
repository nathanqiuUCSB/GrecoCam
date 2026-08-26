"""Queued audio announcements: pre-recorded clips first, TTS fallback."""

import asyncio
import hashlib
import os
import queue
import shutil
import subprocess
import tempfile
import threading

_enabled = True
_backend = "auto"
_voice = "en-US-JennyNeural"
_rate = "+0%"
_volume = "+0%"
_music_mode = False
_sounds_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")

_speak_queue = queue.Queue()
_worker_started = False

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "grecocam_tts_cache")


def configure(enabled=True, backend="auto", voice="en-US-JennyNeural",
              rate="+0%", volume="+0%", sounds_dir=None, music_mode=None):
    global _enabled, _backend, _voice, _rate, _volume, _sounds_dir, _music_mode
    _enabled = enabled
    _backend = backend
    _voice = voice
    _rate = rate
    _volume = volume
    if sounds_dir:
        _sounds_dir = sounds_dir
    if music_mode is not None:
        _music_mode = music_mode
    elif os.environ.get("MUSIC_MODE", "").lower() in ("1", "true", "yes"):
        _music_mode = True


def _resolve_backend():
    if _backend != "auto":
        return _backend
    try:
        import edge_tts  # noqa: F401
        return "edge"
    except ImportError:
        pass
    try:
        import pyttsx3  # noqa: F401
        return "pyttsx3"
    except ImportError:
        pass
    if shutil.which("espeak"):
        return "espeak"
    return "print"


def _clip_path(name):
    """Return sounds clip for name. Music mode prefers <name>music.mp3."""
    if not name or not os.path.isdir(_sounds_dir):
        return None
    stem = name.strip().lower()
    stems = [stem + "music", stem] if _music_mode else [stem]
    for candidate in stems:
        for ext in (".mp3", ".wav", ".ogg"):
            path = os.path.join(_sounds_dir, candidate + ext)
            if os.path.isfile(path):
                return path
        for filename in os.listdir(_sounds_dir):
            base, ext = os.path.splitext(filename)
            if base.lower() == candidate and ext.lower() in (".mp3", ".wav", ".ogg"):
                return os.path.join(_sounds_dir, filename)
    return None


def music_mode_enabled():
    return _music_mode


def announce(name, fallback_text=None):
    """Play sounds/<name>.mp3 if present, otherwise TTS fallback_text or name."""
    if not _enabled:
        print(f"[ANNOUNCE] {fallback_text or name}")
        return
    path = _clip_path(name)
    if path:
        _ensure_worker()
        _speak_queue.put({"type": "file", "path": path, "label": name})
        return
    speak(fallback_text or f"Welcome home, {name}!")


def speak(text):
    if not _enabled:
        print(f"[ANNOUNCE] {text}")
        return
    _ensure_worker()
    _speak_queue.put({"type": "tts", "text": text})


def _ensure_worker():
    global _worker_started
    if not _worker_started:
        threading.Thread(target=_worker_loop, daemon=True).start()
        _worker_started = True


def _worker_loop():
    import pygame

    pygame.mixer.init()
    backend = _resolve_backend()
    while True:
        item = _speak_queue.get()
        try:
            if item["type"] == "file":
                _play_file(pygame, item["path"])
            else:
                _play_tts(pygame, item["text"], backend)
        except Exception as exc:
            print(f"Audio error: {exc}")
            if item.get("type") == "tts":
                _fallback_speak(item["text"])
            elif item.get("label"):
                _fallback_speak(f"Welcome home, {item['label']}!")
        finally:
            _speak_queue.task_done()


def _play_file(pygame, path):
    print(f"[AUDIO] Playing clip: {os.path.basename(path)}")
    sound = pygame.mixer.Sound(path)
    channel = sound.play()
    while channel.get_busy():
        pygame.time.wait(50)
    del sound


def _play_tts(pygame, text, backend):
    if backend == "edge":
        path = _edge_to_file(text)
        sound = pygame.mixer.Sound(path)
        channel = sound.play()
        while channel.get_busy():
            pygame.time.wait(50)
        del sound
        return
    if backend == "pyttsx3":
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return
    if backend == "espeak":
        subprocess.run(["espeak", text], check=False)
        return
    print(f"[ANNOUNCE] {text}")


def _cache_path(text):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    key = hashlib.sha1(f"{_voice}|{_rate}|{_volume}|{text}".encode()).hexdigest()
    return os.path.join(_CACHE_DIR, f"{key}.mp3")


async def _edge_save(text, path):
    import edge_tts

    try:
        communicate = edge_tts.Communicate(text, _voice, rate=_rate, volume=_volume)
        await communicate.save(path)
    except Exception:
        communicate = edge_tts.Communicate(text, _voice)
        await communicate.save(path)


def _edge_to_file(text, retries=3):
    path = _cache_path(text)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            asyncio.run(_edge_save(text, path))
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return path
            last_err = RuntimeError("empty audio file")
        except Exception as exc:
            last_err = exc
            print(f"TTS edge attempt {attempt}/{retries} failed: {exc}")
    raise last_err


def _fallback_speak(text):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return
    except Exception:
        pass
    if shutil.which("espeak"):
        subprocess.run(["espeak", text], check=False)
        return
    print(f"[ANNOUNCE] {text}")
