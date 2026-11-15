# Log triage notes

One entry per triage (newest last): what the logs showed, and what was done.
Made with `/log-triage`; the next triage starts from the date of the last entry.

## 2026-11-15 — two-channel window, Friend and the SPDIF devices

**Period:** 11:35–11:44 (one app run, branch `dual-channel`).
**Reported:** Channel 2 (Friend) won't connect to one SPDIF device; another works.

### Counts (interpreter.error.log.jsonl)

| Count | Level    | Event                 | Source                                   |
|------:|----------|-----------------------|------------------------------------------|
| 26    | WARNING  | source_silent         | interpreter:_report_source (24 Me, 2 Friend) |
| 1     | WARNING  | channel_not_listening | interpreter_qt:_channel_problem          |
| 1     | CRITICAL | unhandled_exception   | interpreter:_excepthook                  |

### Findings

1. **"Digital Output (Realtek USB2.0 Audio)" is not in Windows' device list.**
   It was the Friend device saved by the old mode design. The app never tried
   to open it; it waited. The other SPDIF device, "SPDIF Interface (Realtek
   USB Audio) [Loopback]" (index 33, 44.1 kHz, 2 ch), worked in three runs:
   89–100 % speech, level about −12 dBFS. The capture code works with SPDIF;
   the problem is the device's presence or name. The menu kept the missing
   device as if it were a normal choice, so it looked like a broken device.
2. **CRITICAL `_ignore_signal_value() missing 1 required positional argument`.**
   Fluent buttons send `clicked` with no value; the wrapper required one, so
   the strips' Speak and Mute buttons crashed. The same kind of error
   (`_select_mode`, old mode buttons) was logged three times on 2026-11-09.
3. **24 of 28 warnings were the Me microphone's pauses.** The RODECaster's
   noise gate sends exact silence between words, which the "no app is
   playing" check (meant for loopback devices) reported every 5 s.
4. While Friend waited for the missing device, the watchdog re-scanned all
   devices every 4 s (an `audio_devices_enumerated` line each time, and
   PortAudio restarted while Me was recording).
5. The device menus listed MME copies with names cut to 31 characters
   ("SPDIF Interface (Realtek USB Au", "Microphone (RODECaster Duo Main").
6. `log_tools.py report` mixed the channels: a Friend sensitivity change was
   compared with the Me microphone's level (−89.8 dBFS instead of −9.2).

### Fixed (all on `dual-channel`)

- `_ignore_signal_value(*_, callback)` accepts any number of signal values.
  Verified with real clicks on the strip buttons.
- Missing device: the menu marks it "(not connected)" and drops it once
  another device is chosen; the strip says what to do and names a similar
  connected device (never switches by itself); the log line lists `similar`
  and `available` devices; re-scans back off 3 → 6 → 12 → 24 → 30 s.
- Microphones: silence reported only after 60 s (`MIC_SILENT_SEC`), as INFO.
- MME duplicates with truncated names are left out of the device lists
  (14 → 11 inputs, 10 → 7 outputs here).
- `log_tools.py`: capture levels, speech share, sensitivity before/after,
  the what-if table and "Source silent" are now per channel.

### Recommendation for next time

Check in Windows Sound settings (Win+R, `mmsys.cpl`) whether the Realtek
USB2.0 device still exists (Playback tab, right-click → Show Disabled
Devices). If it is the same hardware under a new name, choose "SPDIF
Interface (Realtek USB Audio)" for Friend and the problem is gone.
