"""Preview TTS voices. Requires: pip install edge-tts pygame

Examples:
  python preview_voices.py
  python preview_voices.py en-US-GuyNeural
  python preview_voices.py en-US-JennyNeural "Welcome home, Nathan!"
"""

import asyncio
import sys

import edge_tts


DEFAULT_PHRASE = "Welcome home, Nathan! Unknown person at the door."

RECOMMENDED = [
    ("en-US-JennyNeural", "friendly female"),
    ("en-US-AriaNeural", "clear female"),
    ("en-US-GuyNeural", "casual male"),
    ("en-US-ChristopherNeural", "deeper male"),
    ("en-GB-SoniaNeural", "British female"),
    ("en-GB-RyanNeural", "British male"),
]


async def list_english_voices():
    voices = await edge_tts.list_voices()
    return [v for v in voices if v["Locale"].startswith("en-") and "Neural" in v["ShortName"]]


async def preview(voice, phrase):
    import pygame
    import os
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "grecocam_preview.mp3")
    await edge_tts.Communicate(phrase, voice).save(path)

    pygame.mixer.init()
    sound = pygame.mixer.Sound(path)
    print(f"Playing {voice}: {phrase}")
    channel = sound.play()
    while channel.get_busy():
        pygame.time.wait(50)


async def main():
    if len(sys.argv) == 1:
        print("Recommended voices:\n")
        for name, desc in RECOMMENDED:
            print(f"  {name}  ({desc})")
        print(f'\nTry: python preview_voices.py en-US-JennyNeural "{DEFAULT_PHRASE}"')
        return

    voice = sys.argv[1]
    phrase = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PHRASE
    await preview(voice, phrase)


if __name__ == "__main__":
    asyncio.run(main())
