"use client";

import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Bell,
  CheckCircle2,
  ChevronRight,
  CircleGauge,
  Clock3,
  Database,
  FileSearch,
  Fingerprint,
  Gauge,
  Menu,
  Radar,
  RefreshCw,
  Search,
  ShieldAlert,
  ShieldCheck,
  SlidersHorizontal,
  XCircle,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip as ChartTooltip,
  XAxis,
  YAxis,
} from "recharts";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Toaster } from "@/components/ui/sonner";

type ReviewDecision = "pending" | "confirmed_fraud" | "cleared";

type ReviewItem = {
  id: string;
  merchant: string;
  channel: string;
  amount: number;
  probability: number;
  country: string;
  time: string;
  reason: string;
  status: ReviewDecision;
};

const riskTrend = [
  { time: "00:00", scored: 428, alerts: 2 },
  { time: "02:00", scored: 312, alerts: 4 },
  { time: "04:00", scored: 267, alerts: 3 },
  { time: "06:00", scored: 458, alerts: 4 },
  { time: "08:00", scored: 742, alerts: 7 },
  { time: "10:00", scored: 890, alerts: 10 },
  { time: "12:00", scored: 975, alerts: 8 },
  { time: "14:00", scored: 1088, alerts: 12 },
  { time: "16:00", scored: 1236, alerts: 9 },
  { time: "18:00", scored: 1104, alerts: 15 },
  { time: "20:00", scored: 864, alerts: 11 },
  { time: "22:00", scored: 638, alerts: 6 },
];

const driftSignals = [
  { feature: "fraud_probability", psi: 0.083 },
  { feature: "Amount", psi: 0.071 },
  { feature: "V14", psi: 0.064 },
  { feature: "V4", psi: 0.052 },
];

const initialReviews: ReviewItem[] = [
  {
    id: "3cc76c35-012a-413b-8770-453aab3a642c",
    merchant: "Asteria Digital",
    channel: "Card not present",
    amount: 1840.5,
    probability: 0.9603,
    country: "NL",
    time: "18:42:16",
    reason: "High model probability",
    status: "pending",
  },
  {
    id: "d2a3d021-5088-4ca6-8b6c-857a7eb11c28",
    merchant: "Northstar Travel",
    channel: "Card not present",
    amount: 967.24,
    probability: 0.8471,
    country: "CA",
    time: "18:39:02",
    reason: "Amount and feature shift",
    status: "pending",
  },
  {
    id: "0ab845b5-c7bf-43de-a776-e255c14b91ed",
    merchant: "Kanso Market",
    channel: "Mobile wallet",
    amount: 486.72,
    probability: 0.7418,
    country: "JP",
    time: "18:34:47",
    reason: "Unusual feature combination",
    status: "pending",
  },
  {
    id: "65c900c5-0701-4525-9765-ce9ee6101179",
    merchant: "Atlas Electronics",
    channel: "Card not present",
    amount: 2260,
    probability: 0.6942,
    country: "MA",
    time: "18:27:31",
    reason: "High transaction amount",
    status: "pending",
  },
  {
    id: "f5a2a763-acb5-4f9f-bc7d-4aa262ef0a31",
    merchant: "Vela Services",
    channel: "Recurring",
    amount: 312.18,
    probability: 0.5834,
    country: "US",
    time: "18:21:09",
    reason: "Prediction above threshold",
    status: "pending",
  },
  {
    id: "487eae31-93ce-4bed-aab2-932bedc99ba7",
    merchant: "Casa Forma",
    channel: "E-commerce",
    amount: 729.96,
    probability: 0.4318,
    country: "ES",
    time: "18:16:52",
    reason: "Prediction above threshold",
    status: "pending",
  },
];

const navItems = [
  { label: "Overview", target: "overview", icon: BarChart3 },
  { label: "Review queue", target: "review-queue", icon: FileSearch, count: true },
  { label: "Model monitor", target: "drift-monitor", icon: Radar },
  { label: "Audit context", target: "review-queue", icon: Fingerprint },
];

const formatMoney = (amount: number) =>
  new Intl.NumberFormat("en-CA", {
    style: "currency",
    currency: "CAD",
    maximumFractionDigits: 2,
  }).format(amount);

function ProbabilityBadge({ probability }: { probability: number }) {
  const level = probability >= 0.8 ? "critical" : probability >= 0.6 ? "high" : "medium";
  const classes = {
    critical: "border-red-400/20 bg-red-400/10 text-red-300",
    high: "border-amber-300/20 bg-amber-300/10 text-amber-200",
    medium: "border-blue-300/20 bg-blue-300/10 text-blue-200",
  };

  return (
    <Badge variant="outline" className={`${classes[level]} data-number px-2.5 py-1`}>
      {(probability * 100).toFixed(1)}%
    </Badge>
  );
}

function MetricCard({
  label,
  value,
  detail,
  icon: Icon,
  accent = "mint",
}: {
  label: string;
  value: string;
  detail: string;
  icon: typeof Activity;
  accent?: "mint" | "amber" | "blue" | "red";
}) {
  const accentClasses = {
    mint: "bg-emerald-300/10 text-emerald-300",
    amber: "bg-amber-300/10 text-amber-200",
    blue: "bg-blue-300/10 text-blue-200",
    red: "bg-red-400/10 text-red-300",
  };

  return (
    <article className="glass-panel rounded-xl border p-4 sm:p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm text-muted-foreground">{label}</p>
          <p className="data-number mt-3 text-2xl font-semibold text-foreground sm:text-[1.75rem]">
            {value}
          </p>
        </div>
        <div className={`rounded-lg p-2.5 ${accentClasses[accent]}`}>
          <Icon className="size-5" aria-hidden="true" />
        </div>
      </div>
      <p className="mt-3 text-xs leading-5 text-muted-foreground">{detail}</p>
    </article>
  );
}

export default function Home() {
  const [reviews, setReviews] = useState(initialReviews);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activeNav, setActiveNav] = useState("Overview");
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [lastUpdated, setLastUpdated] = useState("just now");

  const pendingReviews = useMemo(
    () =>
      reviews.filter(
        (item) =>
          item.status === "pending" &&
          `${item.merchant} ${item.id} ${item.country}`.toLowerCase().includes(query.toLowerCase()),
      ),
    [query, reviews],
  );
  const pendingCount = reviews.filter((item) => item.status === "pending").length;
  const selected = reviews.find((item) => item.id === selectedId) ?? null;

  function navigate(label: string, target: string) {
    setActiveNav(label);
    setMobileNavOpen(false);
    document.getElementById(target)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function resolveReview(decision: Exclude<ReviewDecision, "pending">) {
    if (!selected) return;
    setReviews((items) =>
      items.map((item) => (item.id === selected.id ? { ...item, status: decision } : item)),
    );
    setSelectedId(null);
    toast.success(decision === "confirmed_fraud" ? "Fraud confirmed" : "Transaction cleared", {
      description: `${selected.merchant} · ${formatMoney(selected.amount)}`,
    });
  }

  function refreshData() {
    setLastUpdated("just now");
    toast.success("Monitoring snapshot refreshed");
  }

  const navigation = (
    <nav className="space-y-1" aria-label="Dashboard sections">
      {navItems.map((item) => {
        const Icon = item.icon;
        const active = activeNav === item.label;
        return (
          <button
            key={item.label}
            type="button"
            onClick={() => navigate(item.label, item.target)}
            className={`flex min-h-10 w-full items-center gap-3 rounded-lg px-3 text-left text-sm transition-colors ${
              active
                ? "bg-sidebar-accent text-sidebar-accent-foreground"
                : "text-muted-foreground hover:bg-sidebar-accent/60 hover:text-foreground"
            }`}
          >
            <Icon className={`size-4 ${active ? "text-emerald-300" : ""}`} aria-hidden="true" />
            <span className="flex-1">{item.label}</span>
            {item.count ? (
              <span className="data-number rounded-full bg-amber-300/10 px-2 py-0.5 text-xs text-amber-200">
                {pendingCount}
              </span>
            ) : null}
          </button>
        );
      })}
    </nav>
  );

  return (
    <div className="min-h-screen bg-background/80 text-foreground">
      <Toaster position="bottom-right" richColors />

      <aside className="fixed inset-y-0 left-0 z-30 hidden w-64 border-r bg-sidebar/95 px-4 py-5 backdrop-blur-xl lg:flex lg:flex-col">
        <div className="flex h-12 items-center gap-3 px-2">
          <div className="relative grid size-9 place-items-center rounded-xl border border-emerald-300/20 bg-emerald-300/10 text-emerald-300">
            <Activity className="size-5" aria-hidden="true" />
            <span className="signal-dot absolute -right-0.5 -top-0.5 size-1.5 rounded-full bg-emerald-300" />
          </div>
          <div>
            <p className="text-base font-semibold tracking-tight">RiskPulse</p>
            <p className="text-xs text-muted-foreground">Operations</p>
          </div>
        </div>
        <div className="mt-9">{navigation}</div>

        <div className="mt-auto rounded-xl border bg-card/70 p-4">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Database className="size-4 text-emerald-300" aria-hidden="true" />
            Demo workspace
          </div>
          <p className="mt-2 text-xs leading-5 text-muted-foreground">
            Representative monitoring signals. Connect a deployed API for live data.
          </p>
        </div>

        <div className="mt-4 flex items-center gap-3 border-t px-2 pt-4">
          <div className="grid size-8 place-items-center rounded-full bg-gradient-to-br from-emerald-300 to-cyan-500 text-xs font-bold text-slate-950">
            YJ
          </div>
          <div className="min-w-0">
            <p className="truncate text-sm font-medium">Yasser Jabari</p>
            <p className="truncate text-xs text-muted-foreground">Platform administrator</p>
          </div>
        </div>
      </aside>

      <main className="lg:pl-64">
        <header className="sticky top-0 z-20 flex min-h-16 items-center justify-between border-b bg-background/90 px-4 backdrop-blur-xl sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              className="lg:hidden"
              aria-label="Open navigation"
              onClick={() => setMobileNavOpen(true)}
            >
              <Menu />
            </Button>
            <div>
              <h1 className="text-base font-semibold sm:text-lg">Fraud operations</h1>
              <p className="hidden text-xs text-muted-foreground sm:block">
                Decisioning and model health in one view
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="hidden items-center gap-2 rounded-full border bg-card/70 px-3 py-1.5 text-xs text-muted-foreground md:flex">
              <span className="signal-dot size-1.5 rounded-full bg-emerald-300" />
              All systems operational
            </div>
            <Button variant="ghost" size="icon" aria-label="Notifications">
              <Bell />
            </Button>
            <Button variant="outline" size="sm" onClick={refreshData}>
              <RefreshCw />
              <span className="hidden sm:inline">Refresh</span>
            </Button>
          </div>
        </header>

        <div className="mx-auto max-w-[1600px] space-y-6 px-4 py-5 sm:px-6 lg:px-8 lg:py-7">
          <section id="overview" className="scroll-mt-24 flex flex-col justify-between gap-3 md:flex-row md:items-end">
            <div>
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="border-emerald-300/20 bg-emerald-300/5 text-emerald-200">
                  Demo stream
                </Badge>
                <span className="text-xs text-muted-foreground">Updated {lastUpdated}</span>
              </div>
              <h2 className="mt-3 text-2xl font-semibold tracking-tight sm:text-3xl">Today&apos;s risk posture</h2>
            </div>
            <p className="max-w-xl text-sm leading-6 text-muted-foreground">
              Model <span className="font-mono text-foreground/80">creditcard-hgb-sigmoid-20260727</span>
              <span className="mx-2 text-border">/</span>
              threshold <span className="font-mono text-foreground/80">0.0735</span>
            </p>
          </section>

          <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4" aria-label="Key metrics">
            <MetricCard label="Transactions scored" value="14,208" detail="1,034 in the last hour" icon={Activity} />
            <MetricCard
              label="Manual-review alerts"
              value="17"
              detail="0.12% of scored transactions"
              icon={ShieldAlert}
              accent="amber"
            />
            <MetricCard
              label="Pending review"
              value={String(pendingCount)}
              detail="Oldest item waiting 26 minutes"
              icon={Clock3}
              accent="red"
            />
            <MetricCard
              label="Scoring latency p95"
              value="41 ms"
              detail="Within the 100 ms service target"
              icon={Gauge}
              accent="blue"
            />
          </section>

          <section className="grid gap-4 xl:grid-cols-[minmax(0,1.65fr)_minmax(320px,0.85fr)]">
            <article className="glass-panel min-w-0 rounded-xl border p-4 sm:p-5">
              <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
                <div>
                  <p className="text-sm font-medium">Scoring volume</p>
                  <p className="mt-1 text-xs text-muted-foreground">Transactions processed across the last 24 hours</p>
                </div>
                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                  <span className="flex items-center gap-2"><span className="size-2 rounded-full bg-emerald-300" /> Scored</span>
                  <span className="flex items-center gap-2"><span className="size-2 rounded-full bg-amber-300" /> Alerts</span>
                </div>
              </div>
              <div className="mt-5 h-[260px] w-full" aria-label="Scoring volume chart">
                <ResponsiveContainer
                  width="100%"
                  height="100%"
                  minWidth={0}
                  initialDimension={{ width: 900, height: 260 }}
                >
                  <AreaChart data={riskTrend} margin={{ top: 8, right: 8, left: -22, bottom: 0 }}>
                    <defs>
                      <linearGradient id="scoredGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#54e6c5" stopOpacity={0.32} />
                        <stop offset="100%" stopColor="#54e6c5" stopOpacity={0.01} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="#1d3037" strokeDasharray="3 5" vertical={false} />
                    <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fill: "#78908a", fontSize: 12 }} />
                    <YAxis yAxisId="volume" axisLine={false} tickLine={false} tick={{ fill: "#78908a", fontSize: 12 }} />
                    <YAxis yAxisId="alerts" orientation="right" hide domain={[0, 20]} />
                    <ChartTooltip
                      cursor={{ stroke: "#31535a", strokeDasharray: "4 4" }}
                      contentStyle={{ background: "#0d1b21", border: "1px solid #254047", borderRadius: 10, color: "#e9f2ef", fontSize: 12 }}
                    />
                    <Area yAxisId="volume" type="monotone" dataKey="scored" stroke="#54e6c5" strokeWidth={2} fill="url(#scoredGradient)" />
                    <Area yAxisId="alerts" type="monotone" dataKey="alerts" stroke="#f4b860" strokeWidth={2} fill="transparent" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </article>

            <article id="drift-monitor" className="glass-panel scroll-mt-24 rounded-xl border p-4 sm:p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-sm font-medium">Drift monitor</p>
                  <p className="mt-1 text-xs text-muted-foreground">Population Stability Index · last 1,000 events</p>
                </div>
                <Badge variant="outline" className="border-emerald-300/20 bg-emerald-300/5 text-emerald-200">
                  <ShieldCheck /> Stable
                </Badge>
              </div>
              <div className="mt-6 space-y-5">
                {driftSignals.map((signal) => (
                  <div key={signal.feature}>
                    <div className="mb-2 flex items-center justify-between gap-4">
                      <span className="truncate font-mono text-xs text-foreground/85">{signal.feature}</span>
                      <span className="data-number text-xs text-muted-foreground">PSI {signal.psi.toFixed(3)}</span>
                    </div>
                    <Progress
                      value={(signal.psi / 0.25) * 100}
                      aria-label={`${signal.feature} PSI ${signal.psi}`}
                      className="h-1.5 bg-secondary [&_[data-slot=progress-indicator]]:bg-emerald-300"
                    />
                  </div>
                ))}
              </div>
              <div className="mt-6 grid grid-cols-2 gap-3 border-t pt-4">
                <div><p className="data-number text-lg font-semibold">0.10</p><p className="text-xs text-muted-foreground">Warning level</p></div>
                <div><p className="data-number text-lg font-semibold">0.25</p><p className="text-xs text-muted-foreground">Critical level</p></div>
              </div>
            </article>
          </section>

          <section id="review-queue" className="glass-panel scroll-mt-24 overflow-hidden rounded-xl border">
            <div className="flex flex-col justify-between gap-4 border-b px-4 py-4 sm:flex-row sm:items-center sm:px-5">
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-base font-semibold">Manual-review queue</h2>
                  <Badge variant="secondary" className="data-number bg-amber-300/10 text-amber-200">
                    {pendingCount} pending
                  </Badge>
                </div>
                <p className="mt-1 text-xs text-muted-foreground">Highest probability first</p>
              </div>
              <label className="relative block w-full sm:w-72">
                <span className="sr-only">Search the review queue</span>
                <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                <input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search merchant or transaction"
                  className="h-9 w-full rounded-lg border bg-background/70 pl-9 pr-3 text-sm outline-none placeholder:text-muted-foreground focus:border-emerald-300/50 focus:ring-2 focus:ring-emerald-300/10"
                />
              </label>
            </div>

            {pendingReviews.length > 0 ? (
              <Table>
                <TableHeader>
                  <TableRow className="border-border/80 hover:bg-transparent">
                    <TableHead className="h-11 pl-5 text-xs text-muted-foreground">Transaction</TableHead>
                    <TableHead className="h-11 text-xs text-muted-foreground">Amount</TableHead>
                    <TableHead className="h-11 text-xs text-muted-foreground">Probability</TableHead>
                    <TableHead className="h-11 text-xs text-muted-foreground">Market</TableHead>
                    <TableHead className="h-11 text-xs text-muted-foreground">Received</TableHead>
                    <TableHead className="h-11 pr-5 text-right text-xs text-muted-foreground">Review</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {pendingReviews.map((item) => (
                    <TableRow key={item.id} className="border-border/70 hover:bg-emerald-300/[0.025]">
                      <TableCell className="max-w-[260px] py-3.5 pl-5">
                        <div className="truncate text-sm font-medium">{item.merchant}</div>
                        <div className="mt-1 truncate font-mono text-xs text-muted-foreground">{item.id.slice(0, 18)}…</div>
                      </TableCell>
                      <TableCell className="data-number py-3.5 text-sm font-medium">{formatMoney(item.amount)}</TableCell>
                      <TableCell className="py-3.5"><ProbabilityBadge probability={item.probability} /></TableCell>
                      <TableCell className="py-3.5 text-sm text-muted-foreground">{item.country}</TableCell>
                      <TableCell className="data-number py-3.5 text-sm text-muted-foreground">{item.time}</TableCell>
                      <TableCell className="py-3.5 pr-5 text-right">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setSelectedId(item.id)}
                          aria-label={`Review transaction from ${item.merchant}`}
                          className="text-emerald-200 hover:bg-emerald-300/10 hover:text-emerald-100"
                        >
                          Open <ChevronRight />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : (
              <div className="grid min-h-52 place-items-center px-6 py-12 text-center">
                <div>
                  <CheckCircle2 className="mx-auto size-7 text-emerald-300" />
                  <p className="mt-3 text-sm font-medium">Queue is clear</p>
                  <p className="mt-1 text-xs text-muted-foreground">No pending transactions match this view.</p>
                </div>
              </div>
            )}
          </section>

          <footer className="flex flex-col justify-between gap-2 border-t py-3 text-xs text-muted-foreground sm:flex-row">
            <span>RiskPulse model telemetry · portfolio demonstration</span>
            <span className="font-mono">reference rows: 199,364 · signals: 31</span>
          </footer>
        </div>
      </main>

      <Sheet open={mobileNavOpen} onOpenChange={setMobileNavOpen}>
        <SheetContent side="left" className="w-[86%] border-r bg-[#081419] p-5 sm:max-w-xs">
          <SheetHeader className="px-0 text-left">
            <SheetTitle className="flex items-center gap-3">
              <span className="grid size-9 place-items-center rounded-xl border border-emerald-300/20 bg-emerald-300/10 text-emerald-300">
                <Activity className="size-5" />
              </span>
              RiskPulse Operations
            </SheetTitle>
            <SheetDescription>Fraud decisioning and model health</SheetDescription>
          </SheetHeader>
          <div className="mt-7">{navigation}</div>
        </SheetContent>
      </Sheet>

      <Sheet open={selected !== null} onOpenChange={(open) => !open && setSelectedId(null)}>
        <SheetContent className="w-full overflow-y-auto border-l bg-[#081419] p-0 sm:max-w-lg">
          {selected ? (
            <>
              <SheetHeader className="border-b px-5 py-5 text-left sm:px-6">
                <div className="mb-4 flex items-center gap-2">
                  <Badge variant="outline" className="border-amber-300/20 bg-amber-300/10 text-amber-200">
                    <AlertTriangle /> Manual review
                  </Badge>
                  <span className="text-xs text-muted-foreground">Received {selected.time}</span>
                </div>
                <SheetTitle className="text-xl">{selected.merchant}</SheetTitle>
                <SheetDescription className="font-mono text-xs">{selected.id}</SheetDescription>
              </SheetHeader>

              <div className="space-y-6 px-5 py-6 sm:px-6">
                <section className="risk-grid rounded-xl border bg-card/60 p-5">
                  <div className="flex items-end justify-between gap-4">
                    <div>
                      <p className="text-xs uppercase tracking-[0.16em] text-muted-foreground">Fraud probability</p>
                      <p className="data-number mt-2 text-4xl font-semibold">{(selected.probability * 100).toFixed(2)}%</p>
                    </div>
                    <CircleGauge className="size-9 text-amber-200" aria-hidden="true" />
                  </div>
                  <Progress
                    value={selected.probability * 100}
                    aria-label={`Fraud probability ${(selected.probability * 100).toFixed(2)} percent`}
                    className="mt-5 h-2 bg-secondary [&_[data-slot=progress-indicator]]:bg-amber-300"
                  />
                  <p className="mt-3 text-xs text-muted-foreground">
                    Decision threshold <span className="font-mono text-foreground/80">7.35%</span>
                  </p>
                </section>

                <section>
                  <h3 className="text-sm font-medium">Transaction details</h3>
                  <dl className="mt-3 divide-y rounded-xl border bg-card/40 px-4">
                    {[
                      ["Amount", formatMoney(selected.amount)],
                      ["Market", selected.country],
                      ["Channel", selected.channel],
                      ["Route", "manual_review"],
                      ["Model", "creditcard-hgb-sigmoid-20260727"],
                    ].map(([label, value]) => (
                      <div key={label} className="flex items-center justify-between gap-4 py-3 text-sm">
                        <dt className="text-muted-foreground">{label}</dt>
                        <dd className="max-w-[65%] truncate text-right font-mono text-xs text-foreground/90">{value}</dd>
                      </div>
                    ))}
                  </dl>
                </section>

                <section>
                  <h3 className="text-sm font-medium">Decision context</h3>
                  <div className="mt-3 rounded-xl border bg-card/40 p-4">
                    <div className="flex gap-3">
                      <SlidersHorizontal className="mt-0.5 size-4 shrink-0 text-amber-200" />
                      <div>
                        <p className="text-sm">{selected.reason}</p>
                        <p className="mt-1 text-xs leading-5 text-muted-foreground">
                          This alert is based on anonymized PCA features. Review supporting account data before taking action.
                        </p>
                      </div>
                    </div>
                  </div>
                </section>
              </div>

              <div className="sticky bottom-0 mt-auto grid grid-cols-2 gap-3 border-t bg-[#081419]/95 p-5 backdrop-blur sm:p-6">
                <Button variant="outline" size="lg" onClick={() => resolveReview("cleared")}>
                  <XCircle /> Clear
                </Button>
                <Button
                  size="lg"
                  onClick={() => resolveReview("confirmed_fraud")}
                  className="bg-red-400 text-slate-950 hover:bg-red-300"
                >
                  <ShieldAlert /> Confirm fraud
                </Button>
              </div>
            </>
          ) : null}
        </SheetContent>
      </Sheet>
    </div>
  );
}
