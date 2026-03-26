"""Create save states for Super Mario Kart by navigating menus programmatically.

This script boots the ROM, navigates through menus to start a Time Trial race,
and saves the state once racing begins (ext_mode == 0x1C).
"""

import sys
import time
import numpy as np
from pathlib import Path

import retro

INTEGRATION_DIR = str(Path(__file__).parent / "retro_data")
GAME_NAME = "SuperMarioKart-Snes"

# SNES buttons: B Y SELECT START UP DOWN LEFT RIGHT A X L R
NONE    = np.zeros(12, dtype=np.int8)
START   = np.array([0,0,0,1,0,0,0,0,0,0,0,0], dtype=np.int8)
A_BTN   = np.array([0,0,0,0,0,0,0,0,1,0,0,0], dtype=np.int8)
B_BTN   = np.array([1,0,0,0,0,0,0,0,0,0,0,0], dtype=np.int8)
DOWN    = np.array([0,0,0,0,0,1,0,0,0,0,0,0], dtype=np.int8)
UP      = np.array([0,0,0,0,1,0,0,0,0,0,0,0], dtype=np.int8)
LEFT    = np.array([0,0,0,0,0,0,1,0,0,0,0,0], dtype=np.int8)
RIGHT   = np.array([0,0,0,0,0,0,0,1,0,0,0,0], dtype=np.int8)


def press(env, button, frames=1):
    """Press a button for N frames, then release for a few frames."""
    obs = None
    for _ in range(frames):
        obs, _, _, info = env.step(button)
    # Release
    for _ in range(4):
        obs, _, _, info = env.step(NONE)
    return obs, info


def wait(env, frames):
    """Do nothing for N frames."""
    obs = None
    info = {}
    for _ in range(frames):
        obs, _, _, info = env.step(NONE)
    return obs, info


def get_ram(info, key, default=0):
    return info.get(key, default)


def create_state(character_downs=4, track_id=0x07):
    """Navigate menus and save a state at race start.

    character_downs: how many times to press DOWN from Mario to reach desired character.
        0=Mario, 1=Princess, 2=Bowser, 3=Koopa, 4=Toad, 5=DK Jr, 6=Yoshi, 7=Luigi
        Actually SMK character select is a 2x4 grid:
        Mario    Princess  Bowser   Koopa
        Toad     DK Jr     Yoshi    Luigi
        So Koopa is 3 rights from Mario.
    """
    retro.data.Integrations.add_custom_path(INTEGRATION_DIR)

    # Check if ROM is available
    rom_path = Path(__file__).parent / "Super Mario Kart (USA).sfc"
    integration_rom = Path(INTEGRATION_DIR) / GAME_NAME / "rom.sfc"
    if not integration_rom.exists() and rom_path.exists():
        import shutil
        shutil.copy2(rom_path, integration_rom)
        print(f"Copied ROM to {integration_rom}")

    env = retro.make(
        GAME_NAME,
        inttype=retro.data.Integrations.CUSTOM,
        use_restricted_actions=retro.Actions.ALL,
    )

    obs = env.reset()
    print("Game booted. Navigating menus...")

    # Phase 1: Get past the Nintendo/title screen
    # Just press Start repeatedly until something changes
    print("Pressing Start to skip intro...")
    for i in range(20):
        obs, info = wait(env, 30)
        obs, info = press(env, START, 2)

    # By now we should be at the main menu or mode select
    # Let's check game_mode and ext_mode
    print(f"  game_mode={get_ram(info, 'game_mode')}, ext_mode={get_ram(info, 'ext_mode')}")

    # Phase 2: Select 1 Player (should be default) -> press A
    print("Selecting 1 Player...")
    obs, info = press(env, A_BTN, 2)
    obs, info = wait(env, 30)

    # Phase 3: Select Time Trial (press DOWN once from GP, then A)
    print("Selecting Time Trial...")
    obs, info = press(env, DOWN, 2)
    obs, info = wait(env, 10)
    obs, info = press(env, A_BTN, 2)
    obs, info = wait(env, 30)

    print(f"  game_mode={get_ram(info, 'game_mode')}, ext_mode={get_ram(info, 'ext_mode')}")

    # Phase 4: CC class select (if applicable for Time Trial)
    # In Time Trial there's no CC select, might go straight to character
    # Just press A to confirm whatever is shown
    obs, info = press(env, A_BTN, 2)
    obs, info = wait(env, 30)

    # Phase 5: Character select
    # SMK character grid:
    #   Mario    Princess  Bowser   Koopa
    #   Toad     DK Jr     Yoshi    Luigi
    # Start at Mario (top-left). For Koopa Troopa: 3 rights
    print("Selecting Koopa Troopa...")
    for _ in range(3):
        obs, info = press(env, RIGHT, 2)
        obs, info = wait(env, 10)
    obs, info = press(env, A_BTN, 2)
    obs, info = wait(env, 60)

    print(f"  game_mode={get_ram(info, 'game_mode')}, ext_mode={get_ram(info, 'ext_mode')}")

    # Phase 6: Track select (if Time Trial shows track select)
    # Mario Circuit 1 might be default, press A
    obs, info = press(env, A_BTN, 2)
    obs, info = wait(env, 60)

    print(f"  game_mode={get_ram(info, 'game_mode')}, ext_mode={get_ram(info, 'ext_mode')}")

    # Phase 7: Wait for race to start (ext_mode == 0x1C = 28)
    print("Waiting for race to start...")
    max_wait = 600  # 10 seconds at 60fps
    for i in range(max_wait):
        obs, _, _, info = env.step(NONE)
        if info.get("ext_mode", 0) == 0x1C:
            print(f"  Racing mode detected at frame {i}!")
            # Wait a few more frames for the countdown to finish
            for _ in range(180):  # ~3 seconds
                obs, _, _, info = env.step(NONE)
            break
    else:
        # If we haven't found racing mode yet, try more button presses
        print("  Racing mode not found, trying more button presses...")
        for attempt in range(10):
            obs, info = press(env, A_BTN, 2)
            obs, info = wait(env, 60)
            if info.get("ext_mode", 0) == 0x1C:
                print(f"  Racing mode found after extra press {attempt}!")
                for _ in range(180):
                    obs, _, _, info = env.step(NONE)
                break
            obs, info = press(env, START, 2)
            obs, info = wait(env, 60)
            if info.get("ext_mode", 0) == 0x1C:
                print(f"  Racing mode found after START press {attempt}!")
                for _ in range(180):
                    obs, _, _, info = env.step(NONE)
                break

    # Check if we're in racing mode
    if info.get("ext_mode", 0) == 0x1C:
        print(f"SUCCESS: In racing mode!")
        print(f"  course={get_ram(info, 'course'):#04x}")
        print(f"  lap={get_ram(info, 'lap') - 128}")
        print(f"  checkpoint={get_ram(info, 'checkpoint')}")

        # Save state
        state = env.em.get_state()
        state_path = Path(INTEGRATION_DIR) / GAME_NAME / "MarioCircuit1.state"
        with open(state_path, "wb") as f:
            f.write(state)
        print(f"State saved to: {state_path}")
    else:
        print(f"FAILED: Not in racing mode. ext_mode={get_ram(info, 'ext_mode')}")
        print("You may need to create the state manually or adjust the menu navigation.")

        # Save a debug frame
        from PIL import Image
        img = Image.fromarray(obs)
        img.save("/tmp/mario_kart_debug_frame.png")
        print("Debug frame saved to /tmp/mario_kart_debug_frame.png")

    env.close()


if __name__ == "__main__":
    create_state()
