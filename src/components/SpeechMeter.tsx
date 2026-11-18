import { useEffect, useRef } from 'react';
import { OK } from '../lib/config';

interface SpeechMeterProps {
  prob: number;
  threshold: number;
  speaking: boolean;
  active: boolean;
}

const TICKS = [0, 25, 50, 75, 100];

export function SpeechMeter({ prob, threshold, speaking, active }: SpeechMeterProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const shownRef = useRef(0);

  useEffect(() => {
    if (!active) {
      shownRef.current = 0;
    } else {
      shownRef.current = Math.max(prob, shownRef.current * 0.8);
    }
  }, [prob, active]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 300;
    const cssH = 34;
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cssW, cssH);

    const x0 = 4;
    const span = Math.max(20, cssW - 8);
    const shown = active ? shownRef.current : 0;

    ctx.fillStyle = '#2a2d2e';
    ctx.beginPath();
    ctx.roundRect(x0, 2, span, 16, 4);
    ctx.fill();

    if (shown > 0.005) {
      ctx.fillStyle = speaking ? OK : '#2980B9';
      ctx.beginPath();
      ctx.roundRect(x0, 2, span * Math.min(1, shown), 16, 4);
      ctx.fill();
    }

    const xt = x0 + span * Math.max(0, Math.min(1, threshold));
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(xt, 0);
    ctx.lineTo(xt, 20);
    ctx.stroke();

    ctx.fillStyle = '#9AA0A6';
    ctx.font = '10px "Segoe UI", system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const t of TICKS) {
      const rawX = x0 + (span * t) / 100;
      const x = Math.min(Math.max(rawX, 16), cssW - 16);
      ctx.fillText(`${t}%`, x, 20);
    }
    ctx.restore();
  }, [prob, threshold, speaking, active]);

  return <canvas ref={canvasRef} className="block h-[34px] w-full" />;
}
