import { LANGUAGES, SILENCE_FLOOR_DBFS, vadThresholdFor } from './config';
import type { AudioDeviceInfo } from './types';

interface SpeechRecognitionAlternativeLike {
  transcript: string;
  confidence?: number;
}

interface SpeechRecognitionResultLike {
  isFinal: boolean;
  length: number;
  [index: number]: SpeechRecognitionAlternativeLike;
}

interface SpeechRecognitionEventLike extends Event {
  resultIndex: number;
  results: {
    length: number;
    [index: number]: SpeechRecognitionResultLike;
  };
}

interface SpeechRecognitionErrorEventLike extends Event {
  error: string;
  message?: string;
}

interface SpeechRecognitionLike extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((ev: SpeechRecognitionEventLike) => void) | null;
  onerror: ((ev: SpeechRecognitionErrorEventLike) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
}

type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

export function getSpeechRecognitionCtor(): SpeechRecognitionCtor | null {
  const w = window as unknown as {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export async function enumerateWebAudioDevices(): Promise<{
  inputs: AudioDeviceInfo[];
  outputs: AudioDeviceInfo[];
}> {
  const inputs: AudioDeviceInfo[] = [];
  const outputs: AudioDeviceInfo[] = [];

  if (navigator.mediaDevices?.enumerateDevices) {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      let micCount = 0;
      let outCount = 0;
      for (const d of devices) {
        if (d.kind === 'audioinput') {
          micCount += 1;
          inputs.push({
            deviceId: d.deviceId,
            label: d.label || `Microphone ${micCount}`,
            kind: 'audioinput',
          });
        } else if (d.kind === 'audiooutput') {
          outCount += 1;
          outputs.push({
            deviceId: d.deviceId,
            label: d.label || `Speaker / Output ${outCount}`,
            kind: 'audiooutput',
          });
        }
      }
    } catch {
      // Fall back to virtual entries below
    }
  }

  if (inputs.length === 0) {
    inputs.push({
      deviceId: 'default',
      label: 'Default Microphone',
      kind: 'audioinput',
    });
  }
  inputs.push({
    deviceId: '__tab_loopback__',
    label: 'Browser Tab / System Audio [Loopback]',
    kind: 'audioinput',
    isLoopback: true,
  });
  inputs.push({
    deviceId: '__simulated__',
    label: 'Simulated Audio Input (Test & Replay)',
    kind: 'audioinput',
    isSimulated: true,
  });

  if (outputs.length === 0) {
    outputs.push({
      deviceId: 'default',
      label: 'Default Audio Output',
      kind: 'audiooutput',
    });
  }

  return { inputs, outputs };
}

export interface ChannelPipelineCallbacks {
  onLevel: (rmsDb: number, peakDb: number, speaking: boolean, prob: number, threshold: number) => void;
  onStatus: (text: string, colour?: string) => void;
  onPartial: (seq: number, text: string, tStartMs: number) => void;
  onFinal: (seq: number, text: string, tStartMs: number, tEndMs: number, onsetMs: number | null) => void;
  onIssue: (event: string, technical: string) => void;
}

export class WebChannelPipeline {
  private audioCtx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private meterTimer: number | null = null;
  private recognition: SpeechRecognitionLike | null = null;
  private stopped = false;
  private sensitivity = 0.5;
  private isMutedByTts = false;
  private seq = 1;
  private phraseStartMs = 0;
  private firstOnsetMs: number | null = null;
  private wasSpeaking = false;
  private replayGain: GainNode | null = null;

  constructor(
    private readonly deviceLabel: string,
    private readonly devices: AudioDeviceInfo[],
    private readonly srcLanguage: string,
    sensitivityInitial: number,
    private readonly replay: boolean,
    private readonly callbacks: ChannelPipelineCallbacks,
  ) {
    this.sensitivity = sensitivityInitial;
  }

  setSensitivity(value: number): void {
    this.sensitivity = value;
  }

  setTtsMuted(muted: boolean): void {
    this.isMutedByTts = muted;
  }

  async start(): Promise<boolean> {
    this.stopped = false;
    const matched = this.devices.find((d) => d.label === this.deviceLabel);
    const isSimulated = matched?.isSimulated || this.deviceLabel.toLowerCase().includes('simulated');
    const isLoopback = matched?.isLoopback || this.deviceLabel.includes('[Loopback]');

    if (isSimulated) {
      this.callbacks.onStatus('Listening (Simulated input ready — inject phrases below)…');
      this.startSimulatedMeterLoop();
      return true;
    }

    try {
      if (isLoopback) {
        if (!navigator.mediaDevices?.getDisplayMedia) {
          throw new Error('Browser does not support tab/system audio capture (getDisplayMedia).');
        }
        const displayStream = await navigator.mediaDevices.getDisplayMedia({
          audio: true,
          video: true,
        });
        // Stop video tracks immediately; we only need the loopback audio track
        for (const vt of displayStream.getVideoTracks()) {
          vt.stop();
        }
        if (displayStream.getAudioTracks().length === 0) {
          throw new Error('No audio track shared. Enable "Share tab audio" when prompted.');
        }
        this.stream = displayStream;
      } else {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error('Browser does not support microphone capture (getUserMedia).');
        }
        const constraints: MediaStreamConstraints = {
          audio: matched?.deviceId && matched.deviceId !== 'default'
            ? { deviceId: { ideal: matched.deviceId }, echoCancellation: true, noiseSuppression: true }
            : { echoCancellation: true, noiseSuppression: true },
        };
        this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.callbacks.onIssue('mic_permission_denied', msg);
      this.callbacks.onStatus(`⚠ ${msg}`);
      return false;
    }

    if (this.stopped) {
      this.cleanupStream();
      return false;
    }

    this.startAudioMeter(this.stream);
    if (!isLoopback) {
      this.startSpeechRecognition();
    } else {
      this.callbacks.onStatus('Listening to shared tab audio (use Simulator bar for phrase injection if needed)…');
    }
    return true;
  }

  private startSimulatedMeterLoop(): void {
    const tick = () => {
      if (this.stopped) return;
      const threshold = vadThresholdFor(this.sensitivity);
      // Ambient digital silence floor around -62 dBFS
      const noiseDb = -64 + Math.random() * 4;
      this.callbacks.onLevel(noiseDb, noiseDb + 1.5, false, 0.02, threshold);
      this.meterTimer = window.setTimeout(tick, 50);
    };
    tick();
  }

  private startAudioMeter(stream: MediaStream): void {
    const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const ctx = new AudioCtx();
    this.audioCtx = ctx;
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    source.connect(analyser);

    if (this.replay) {
      this.replayGain = ctx.createGain();
      this.replayGain.gain.value = 0.85;
      source.connect(this.replayGain);
      this.replayGain.connect(ctx.destination);
    }

    const buf = new Float32Array(analyser.fftSize);
    const tick = () => {
      if (this.stopped) return;
      analyser.getFloatTimeDomainData(buf);
      let sumSq = 0;
      let peak = 0;
      let zeroCrossings = 0;
      for (let i = 0; i < buf.length; i++) {
        const s = this.isMutedByTts ? 0 : buf[i];
        sumSq += s * s;
        const abs = Math.abs(s);
        if (abs > peak) peak = abs;
        if (i > 0 && ((buf[i - 1] >= 0 && buf[i] < 0) || (buf[i - 1] < 0 && buf[i] >= 0))) {
          zeroCrossings += 1;
        }
      }
      const rms = Math.sqrt(sumSq / buf.length);
      const rmsDb = rms > 1e-6 ? Math.max(-100, 20 * Math.log10(rms)) : -100;
      const peakDb = peak > 1e-6 ? Math.max(-100, 20 * Math.log10(peak)) : -100;

      const threshold = vadThresholdFor(this.sensitivity);
      let prob = 0;
      if (rmsDb > SILENCE_FLOOR_DBFS) {
        // Estimate speech probability from energy above floor & zero-crossing band
        const energyScore = Math.max(0, Math.min(1, (rmsDb + 58) / 36));
        const zcr = zeroCrossings / buf.length;
        const voiceBandFactor = zcr >= 0.015 && zcr <= 0.25 ? 1.0 : 0.6;
        prob = Math.min(1, energyScore * voiceBandFactor);
      }

      const speaking = this.wasSpeaking
        ? prob >= Math.max(0.05, threshold - 0.15)
        : prob >= threshold;

      const now = performance.now();
      if (prob >= 0.2 && this.firstOnsetMs === null) {
        this.firstOnsetMs = now;
      }
      if (speaking && !this.wasSpeaking) {
        this.phraseStartMs = now;
      } else if (!speaking && this.wasSpeaking) {
        this.firstOnsetMs = null;
      }
      this.wasSpeaking = speaking;

      this.callbacks.onLevel(rmsDb, peakDb, speaking, prob, threshold);
      this.meterTimer = window.setTimeout(tick, 32);
    };
    tick();
  }

  private startSpeechRecognition(): void {
    const Ctor = getSpeechRecognitionCtor();
    if (!Ctor) {
      this.callbacks.onIssue(
        'speech_api_unsupported',
        'window.SpeechRecognition is not available in this browser.',
      );
      this.callbacks.onStatus('Listening (meters active; use Simulator bar for text input)');
      return;
    }

    const rec = new Ctor();
    this.recognition = rec;
    rec.continuous = true;
    rec.interimResults = true;

    const entry = LANGUAGES[this.srcLanguage];
    rec.lang = !entry || entry.speechCode === 'auto' ? navigator.language || 'en-US' : entry.speechCode;

    let activeSeq = this.seq;
    let activeStart = performance.now();

    rec.onresult = (event) => {
      if (this.stopped || this.isMutedByTts) return;
      let interim = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const res = event.results[i];
        const transcript = (res[0]?.transcript || '').trim();
        if (!transcript) continue;
        if (res.isFinal) {
          const endMs = performance.now();
          const onsetMs =
            this.firstOnsetMs !== null && this.phraseStartMs > 0
              ? Math.max(0, Math.round(this.phraseStartMs - this.firstOnsetMs))
              : 45;
          this.callbacks.onFinal(activeSeq, transcript, activeStart || endMs - 800, endMs, onsetMs);
          this.seq += 1;
          activeSeq = this.seq;
          activeStart = performance.now();
        } else {
          interim = interim ? `${interim} ${transcript}` : transcript;
        }
      }
      if (interim) {
        if (!activeStart) activeStart = performance.now();
        this.callbacks.onPartial(activeSeq, interim, activeStart);
      }
    };

    rec.onerror = (ev) => {
      if (this.stopped) return;
      if (ev.error === 'no-speech' || ev.error === 'aborted') return;
      if (ev.error === 'not-allowed' || ev.error === 'service-not-allowed') {
        this.callbacks.onIssue('mic_permission_denied', `SpeechRecognition error: ${ev.error}`);
        this.callbacks.onStatus(`⚠ Speech recognition blocked (${ev.error})`);
        return;
      }
      this.callbacks.onStatus(`Speech notice: ${ev.error}`);
    };

    rec.onend = () => {
      if (this.stopped) return;
      // Watchdog auto-restart for continuous recognition
      window.setTimeout(() => {
        if (!this.stopped && this.recognition === rec) {
          try {
            rec.start();
          } catch {
            // Ignore already-started race
          }
        }
      }, 250);
    };

    try {
      rec.start();
      this.callbacks.onStatus('Listening…');
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.callbacks.onStatus(`Speech notice: ${msg}`);
    }
  }

  private cleanupStream(): void {
    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        track.stop();
      }
      this.stream = null;
    }
  }

  stop(): void {
    this.stopped = true;
    if (this.meterTimer !== null) {
      window.clearTimeout(this.meterTimer);
      this.meterTimer = null;
    }
    if (this.recognition) {
      try {
        this.recognition.onend = null;
        this.recognition.stop();
      } catch {
        // Ignore stop errors
      }
      this.recognition = null;
    }
    if (this.audioCtx) {
      this.audioCtx.close().catch(() => {});
      this.audioCtx = null;
    }
    this.cleanupStream();
  }
}
