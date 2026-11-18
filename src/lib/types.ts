export interface ChannelSettings {
  name: string;
  device: string;
  src: string;
  tgt: string;
  sensitivity: number;
  muted: boolean;
  speak: boolean;
  output: string;
  replay: boolean;
}

export interface AppSettings {
  channels: [ChannelSettings, ChannelSettings];
  model: string;
  previewModel: string;
  autoLanguage: boolean;
  textSize: number;
}

export interface AudioDeviceInfo {
  deviceId: string;
  label: string;
  kind: 'audioinput' | 'audiooutput';
  isLoopback?: boolean;
  isSimulated?: boolean;
}

export interface ConversationEntry {
  key: string;
  channel: number;
  time: string;
  t: number;
  heard: string;
  stable: number;
  final: boolean;
  translation: string | null;
  translationFinal: boolean;
  showLabel: boolean;
  detectedLang?: string;
}

export interface PhraseMetric {
  key: string;
  time: string;
  who: string;
  channel: number;
  model: string;
  pauseToTextMs: number;
  firstPreviewMs: number | null;
  processMs: number | null;
  mtMs: number | null;
  early: boolean;
  doubtful: boolean;
  retryUsed: boolean;
  contextReset: boolean;
  onsetMs: number | null;
  text: string;
  translation: string;
  srcLang: string;
  tgtLang: string;
  provider?: string;
}

export interface IssueEntry {
  id: string;
  time: string;
  level: 'INFO' | 'WARNING' | 'ERROR';
  event: string;
  title: string;
  meaning: string;
  action: string;
  technical: string;
  where: string;
  details: string;
}

export interface IssueGroup {
  event: string;
  first: string;
  last: string;
  count: number;
  level: 'WARNING' | 'ERROR';
  issue: IssueEntry;
}

export interface ChannelRuntimeState {
  listening: boolean;
  problem: string | null;
  statusText: string;
  statusColour: string | null;
  rmsDb: number;
  peakDb: number;
  speaking: boolean;
  speechProb: number;
  threshold: number;
  levels: number[];
  speechFlags: boolean[];
  peakRecent: number;
  musicDetected: boolean;
  failures: number;
}

export interface ServerStats {
  nodeVersion: string;
  platform: string;
  arch: string;
  cpus: number;
  rssMb: number;
  heapUsedMb: number;
  uptimeSec: number;
  loadAvg1m: number;
  translationCacheSize: number;
  providers: Record<string, { coolingDownSec: number; cooldownSec: number }>;
}

export interface FeedbackReport {
  id: string;
  timestamp: string;
  channel: string;
  text: string;
  translation: string;
  problems: string[];
  correctText: string;
  actualLanguage: string | null;
  note: string;
  srcLang: string;
  tgtLang: string;
}
