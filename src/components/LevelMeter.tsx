import { useEffect, useRef } from 'react';

interface LevelMeterProps {
  rmsDb: number;
  peakDb: number;
  active: boolean;
}

const FLOOR_DB = -60.0;
const SEG_DB = 1.5;
const FALL_DB_PER_SEC = 20.0;
const HOLD_SEC = 1.5;
const TICKS = [-60, -48, -36, -24, -18, -12, -6, -3, 0];

function segColour(db: number, lit: boolean): string {
  if (db < -18) return lit ? '#2ECC71' : '#123d24';
  if (db < -6) return lit ? '#F1C40F' : '#4a3f0c';
  return lit ? '#E74C3C' : '#4a1712';
}

export function LevelMeter({ rmsDb, peakDb, active }: LevelMeterProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stateRef = useRef({
    rms: FLOOR_DB,
    peak: FLOOR_DB,
    peakT: 0,
    lastT: performance.now() / 1000,
    targetRms: FLOOR_DB,
    targetPeak: FLOOR_DB,
  });

  useEffect(() => {
    const s = stateRef.current;
    if (!active) {
      s.rms = FLOOR_DB;
      s.peak = FLOOR_DB;
      s.targetRms = FLOOR_DB;
      s.targetPeak = FLOOR_DB;
    } else {
      s.targetRms = rmsDb;
      s.targetPeak = peakDb;
    }
  }, [rmsDb, peakDb, active]);

  useEffect(() => {
    let rafId = 0;

    const draw = () => {
      const canvas = canvasRef.current;
      if (canvas) {
        const dpr = window.devicePixelRatio || 1;
        const cssW = canvas.clientWidth || 300;
        const cssH = 34;
        if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
          canvas.width = Math.round(cssW * dpr);
          canvas.height = Math.round(cssH * dpr);
        }
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.save();
          ctx.scale(dpr, dpr);
          ctx.clearRect(0, 0, cssW, cssH);

          const st = stateRef.current;
          const now = performance.now() / 1000;
          const fall = FALL_DB_PER_SEC * Math.max(0, now - st.lastT);
          st.lastT = now;

          st.rms = Math.max(st.targetRms, st.rms - fall);
          if (st.targetPeak >= st.peak) {
            st.peak = st.targetPeak;
            st.peakT = now;
          } else if (now - st.peakT > HOLD_SEC) {
            st.peak = Math.max(st.targetPeak, st.peak - fall);
          }

          const n = Math.floor(-FLOOR_DB / SEG_DB);
          const segW = (cssW - 8) / n;
          const holdIdx = Math.floor((st.peak - FLOOR_DB) / SEG_DB) - 1;

          for (let i = 0; i < n; i++) {
            const db = FLOOR_DB + (i + 1) * SEG_DB;
            const lit = db <= st.rms || i === holdIdx;
            ctx.fillStyle = segColour(db - 0.01, lit);
            ctx.fillRect(4 + i * segW, 2, Math.max(1, segW - 2), 16);
          }

          ctx.fillStyle = '#9AA0A6';
          ctx.font = '10px "Segoe UI", system-ui, sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';
          for (const t of TICKS) {
            const rawX = 4 + ((t - FLOOR_DB) / -FLOOR_DB) * (cssW - 8);
            const x = Math.min(Math.max(rawX, 12), cssW - 12);
            ctx.fillText(String(t), x, 20);
          }
          ctx.restore();
        }
      }
      rafId = requestAnimationFrame(draw);
    };

    rafId = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafId);
  }, []);

  return <canvas ref={canvasRef} className="block h-[34px] w-full" />;
}
