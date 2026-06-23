// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { Alert } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { classifyGrid, GRID_COPY } from '@/lib/grid';
import { runRollout, type RolloutResult } from '@/lib/wasm';

const COLORS = {
  model: '#4f46e5',
  oracle: '#0d9488',
  setpoint: '#94a3b8',
};

export function Demo() {
  const [tau, setTau] = useState(0.65);
  const [gain, setGain] = useState(1.2);
  const [steps, setSteps] = useState(180);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RolloutResult | null>(null);

  // WebGPU is a "powerful feature" gated to secure contexts: https, or
  // http://localhost / 127.0.0.1. A plain-http LAN IP (e.g. the Network URL the
  // dev server prints with --host) is NOT secure, so the browser hides
  // navigator.gpu even when WebGPU is fully supported. Probe in an effect so the
  // server-rendered HTML and the first client render agree (no hydration skew);
  // assume support until the probe runs so the common case shows the controls
  // immediately.
  const [hasWebGPU, setHasWebGPU] = useState(true);
  const [isSecure, setIsSecure] = useState(true);
  useEffect(() => {
    setHasWebGPU(typeof navigator !== 'undefined' && 'gpu' in navigator);
    setIsSecure(typeof window !== 'undefined' && window.isSecureContext);
  }, []);

  const grid = classifyGrid(tau, gain);

  async function onRun() {
    setRunning(true);
    setError(null);
    try {
      const r = await runRollout(tau, gain, steps);
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="shell">
      <Masthead />

      {!hasWebGPU ? (
        <WebGPUNotice insecure={!isSecure} />
      ) : (
        <div className="grid-stack">
          <ControlsCard
            tau={tau}
            gain={gain}
            steps={steps}
            running={running}
            onTau={setTau}
            onGain={setGain}
            onSteps={setSteps}
            onRun={onRun}
            gridKind={grid}
          />

          {error && <ErrorNotice message={error} />}

          {result ? <Results result={result} /> : !error && <EmptyState />}
        </div>
      )}

      <p className="footnote">
        All physics &amp; control run in Rust → WebAssembly → WebGPU via{' '}
        <code>rollout_compare()</code>. Nothing about the plant or PID is
        re-implemented in JavaScript — the page only renders what Rust returns.
      </p>
    </div>
  );
}

/* ----------------------------------------------------------------- header */
function Masthead() {
  return (
    <header className="masthead">
      <span className="eyebrow">
        <span className="dot" />
        Rust · WebAssembly · WebGPU
      </span>
      <h1 className="title">PID Transformer — Model vs. Oracle</h1>
      <p className="lede">
        A tiny transformer learned to imitate a per-plant-tuned PID controller —
        and it generalizes to plant dynamics it never saw. Here both closed
        loops drive the <em>same</em> first-order plant on the <em>same</em>{' '}
        setpoint schedule, so you can watch the model track against the
        analytically-tuned oracle.
      </p>
    </header>
  );
}

/* ---------------------------------------------------------------- controls */
interface ControlsProps {
  tau: number;
  gain: number;
  steps: number;
  running: boolean;
  gridKind: ReturnType<typeof classifyGrid>;
  onTau: (v: number) => void;
  onGain: (v: number) => void;
  onSteps: (v: number) => void;
  onRun: () => void;
}

function ControlsCard(p: ControlsProps) {
  const copy = GRID_COPY[p.gridKind];
  return (
    <Card className="card-pad">
      <div className="controls-head">
        <div>
          <p className="section-label">Plant</p>
          <h2 className="card-title">First-order dynamics</h2>
          <p className="card-sub">
            Set the plant time constant and DC gain, then run both controllers.
          </p>
        </div>
        <Badge kind={p.gridKind} title={copy.hint}>
          <span className="bdot" />
          {copy.label}
        </Badge>
      </div>

      <div className="sliders">
        <SliderRow
          label="τ (time constant)"
          value={p.tau}
          min={0.4}
          max={0.9}
          step={0.01}
          fmt={(v) => v.toFixed(2)}
          disabled={p.running}
          onChange={p.onTau}
        />
        <SliderRow
          label="gain (DC gain)"
          value={p.gain}
          min={0.95}
          max={1.55}
          step={0.005}
          fmt={(v) => v.toFixed(3)}
          disabled={p.running}
          onChange={p.onGain}
        />
        <SliderRow
          label="steps (horizon)"
          value={p.steps}
          min={60}
          max={240}
          step={1}
          fmt={(v) => String(Math.round(v))}
          disabled={p.running}
          onChange={(v) => p.onSteps(Math.round(v))}
        />
      </div>

      <div className="runbar">
        <Button onClick={p.onRun} disabled={p.running}>
          {p.running && <span className="spinner" />}
          {p.running ? 'running…' : 'Run comparison'}
        </Button>
        <span className="run-hint">
          Training grid: τ∈{'{'}0.45, 0.65, 0.85{'}'} · gain∈{'{'}1.00, 1.25, 1.50
          {'}'}. Validation: τ∈{'{'}0.55, 0.75{'}'} · gain∈{'{'}1.125, 1.375{'}'}.
        </span>
      </div>
    </Card>
  );
}

interface SliderRowProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled: boolean;
  fmt: (v: number) => string;
  onChange: (v: number) => void;
}

function SliderRow(p: SliderRowProps) {
  return (
    <div className="slider-block">
      <label>
        <span>{p.label}</span>
        <span className="val">{p.fmt(p.value)}</span>
      </label>
      <Slider
        min={p.min}
        max={p.max}
        step={p.step}
        value={[p.value]}
        disabled={p.disabled}
        onValueChange={(vals) => p.onChange(vals[0])}
      />
      <div className="slider-meta">
        <span>{p.fmt(p.min)}</span>
        <span>{p.fmt(p.max)}</span>
      </div>
    </div>
  );
}

/* ----------------------------------------------------------------- results */
function Results({ result }: { result: RolloutResult }) {
  const trackData = useMemo(
    () =>
      result.t.map((t, i) => ({
        t,
        setpoint: result.setpoint[i],
        model_y: result.model_y[i],
        oracle_y: result.oracle_y[i],
      })),
    [result],
  );

  const ctrlData = useMemo(
    () =>
      result.t.map((t, i) => ({
        t,
        model_u: result.model_u[i],
        oracle_u: result.oracle_u[i],
      })),
    [result],
  );

  const gap = result.model_mse - result.oracle_mse;

  return (
    <>
      <div className="metrics">
        <Metric
          label="Model tracking MSE"
          color={COLORS.model}
          value={result.model_mse}
          foot="transformer-driven loop"
        />
        <Metric
          label="Oracle tracking MSE"
          color={COLORS.oracle}
          value={result.oracle_mse}
          foot="pole-placed velocity-PI"
        />
        <GapMetric gap={gap} />
      </div>

      <ChartCard
        title="Tracking — plant output vs. setpoint"
        subtitle={`τ = ${result.tau.toFixed(2)} · gain = ${result.gain.toFixed(
          3,
        )} · ${result.steps} steps`}
      >
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={trackData} margin={{ top: 8, right: 12, left: -6, bottom: 4 }}>
            <CartesianGrid stroke="#eef1f6" vertical={false} />
            <XAxis
              dataKey="t"
              tick={axisTick}
              stroke="#cbd2dd"
              tickLine={false}
              label={{ value: 'time step', position: 'insideBottom', offset: -2, fill: '#8b93a3', fontSize: 12 }}
            />
            <YAxis tick={axisTick} stroke="#cbd2dd" tickLine={false} width={42} />
            <Tooltip content={<ChartTooltip />} />
            <Legend wrapperStyle={legendStyle} iconType="plainline" />
            <Line
              name="setpoint"
              type="stepAfter"
              dataKey="setpoint"
              stroke={COLORS.setpoint}
              strokeWidth={1.6}
              strokeDasharray="5 4"
              dot={false}
              isAnimationActive={false}
            />
            <Line
              name="model y"
              type="monotone"
              dataKey="model_y"
              stroke={COLORS.model}
              strokeWidth={2.4}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              name="oracle y"
              type="monotone"
              dataKey="oracle_y"
              stroke={COLORS.oracle}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Control signal" subtitle="actuator command u(t) chosen by each controller">
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={ctrlData} margin={{ top: 8, right: 12, left: -6, bottom: 4 }}>
            <CartesianGrid stroke="#eef1f6" vertical={false} />
            <XAxis
              dataKey="t"
              tick={axisTick}
              stroke="#cbd2dd"
              tickLine={false}
              label={{ value: 'time step', position: 'insideBottom', offset: -2, fill: '#8b93a3', fontSize: 12 }}
            />
            <YAxis tick={axisTick} stroke="#cbd2dd" tickLine={false} width={42} />
            <Tooltip content={<ChartTooltip />} />
            <Legend wrapperStyle={legendStyle} iconType="plainline" />
            <Line
              name="model u"
              type="monotone"
              dataKey="model_u"
              stroke={COLORS.model}
              strokeWidth={2.2}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              name="oracle u"
              type="monotone"
              dataKey="oracle_u"
              stroke={COLORS.oracle}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>
    </>
  );
}

const axisTick = { fontSize: 11.5, fill: '#8b93a3' };
const legendStyle = { paddingTop: 8, fontSize: 13 };

function Metric({
  label,
  color,
  value,
  foot,
}: {
  label: string;
  color: string;
  value: number;
  foot: string;
}) {
  return (
    <div className="metric">
      <div className="m-head">
        <span className="swatch" style={{ background: color }} />
        {label}
      </div>
      <div className="m-val">{fmtMse(value)}</div>
      <div className="m-foot">{foot} · lower is better</div>
    </div>
  );
}

function GapMetric({ gap }: { gap: number }) {
  // gap = model − oracle. Negative/near-zero means the model matches or beats the oracle.
  const good = gap <= 1e-4;
  const sign = gap >= 0 ? '+' : '−';
  return (
    <div className="metric gap">
      <div className="m-head">
        <span className="swatch" style={{ background: 'linear-gradient(90deg,#4f46e5,#0d9488)' }} />
        Gap (model − oracle)
      </div>
      <div className={`m-val ${good ? 'good' : 'bad'}`}>
        {sign}
        {fmtMse(Math.abs(gap))}
      </div>
      <div className="m-foot">
        {good ? 'model matches/beats the oracle' : 'oracle ahead by this margin'}
      </div>
    </div>
  );
}

function fmtMse(v: number): string {
  if (v === 0) return '0';
  if (Math.abs(v) < 1e-4) return v.toExponential(2);
  return v.toFixed(5);
}

/* -------------------------------------------------------------- chart bits */
function ChartCard({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <Card className="chart-card">
      <div className="chart-head">
        <div>
          <h3 className="card-title">{title}</h3>
        </div>
        <span className="card-sub">{subtitle}</span>
      </div>
      {children}
    </Card>
  );
}

interface TooltipPayloadItem {
  name?: string;
  value?: number | string;
  color?: string;
}

function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: TooltipPayloadItem[];
  label?: number | string;
}) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="tooltip">
      <div className="tt-label">step {label}</div>
      {payload.map((p) => (
        <div className="tt-row" key={p.name}>
          <span className="tt-sw" style={{ background: p.color }} />
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
        </div>
      ))}
    </div>
  );
}

/* -------------------------------------------------------------- empty/error */
function EmptyState() {
  return (
    <Card className="empty">
      <p className="big">Ready when you are.</p>
      <p>
        Pick a plant with the sliders above and hit <strong>Run comparison</strong>{' '}
        to roll out both closed loops in WebGPU.
      </p>
    </Card>
  );
}

function WebGPUNotice({ insecure }: { insecure: boolean }) {
  return (
    <Alert variant="warn">
      <div className="icon">⚠️</div>
      <div>
        <h3>WebGPU isn’t available in this browser</h3>
        {insecure ? (
          <p>
            This page is on an <strong>insecure origin</strong> (
            <code>{typeof window !== 'undefined' ? window.location.origin : ''}</code>
            ), where browsers hide <code>navigator.gpu</code> even when WebGPU
            works. Open it at <code>http://localhost:5173</code> or{' '}
            <code>http://127.0.0.1:5173</code> instead of a LAN IP — or serve it
            over <code>https://</code>. (<code>localhost</code> counts as secure;
            a bare <code>http://192.168.x.x</code> does not.)
          </p>
        ) : (
          <p>
            This demo runs the controllers on the GPU via WebAssembly, which
            needs <code>navigator.gpu</code>. Open it in Chrome/Edge 113+ (or
            Firefox / Safari with WebGPU enabled), served over a secure origin
            (<code>https://</code> or <code>http://localhost</code>).
          </p>
        )}
      </div>
    </Alert>
  );
}

function ErrorNotice({ message }: { message: string }) {
  return (
    <Alert variant="error">
      <div className="icon">⨯</div>
      <div>
        <h3>Something went wrong</h3>
        <p>
          <code>{message}</code>
        </p>
      </div>
    </Alert>
  );
}
